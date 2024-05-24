import io
import logging

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForZeroShotImageClassification, AutoTokenizer, ZeroShotImageClassificationPipeline
from ultralytics import YOLO

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler()
    ])


class PipelineWithoutPostprocess(ZeroShotImageClassificationPipeline):
    def postprocess(self, model_outputs):
        candidate_labels = model_outputs.pop("candidate_labels")
        logits = model_outputs["logits"][0]
        if self.framework == "pt" and self.model.config.model_type == "siglip":
            probs = torch.sigmoid(logits).squeeze(-1)
            scores = probs.tolist()
            if not isinstance(scores, list):
                scores = [scores]
        elif self.framework == "pt":
            # probs = logits.softmax(dim=-1).squeeze(-1)
            probs = logits.squeeze(-1)  # no softmax because only 1 target class at test time, softmax causes it to go 1.0 for all
            scores = probs.tolist()
            if not isinstance(scores, list):
                scores = [scores]
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        return result


class VLMManager:
    def __init__(self, yolo_path: str, clip_path: str):
        logging.info(f'Loading YOLO model from {yolo_path}')
        self.yolo_model = YOLO(yolo_path)
        logging.info(f'Loading CLIP model from {clip_path}')
        self.clip_model = PipelineWithoutPostprocess(task="zero-shot-image-classification",
                                                     model=AutoModelForZeroShotImageClassification.from_pretrained(clip_path),
                                                     tokenizer=AutoTokenizer.from_pretrained(clip_path),
                                                     image_processor=AutoImageProcessor.from_pretrained(clip_path),
                                                     batch_size=4, device='cuda')
        logging.info('VLMManager initialized')

    def identify(self, img_bytes: list[bytes], captions: list[str]) -> list[list[int]]:
        logging.info('Predicting')
        # image is the raw bytes of a JPEG file
        images = [Image.open(io.BytesIO(b)) for b in img_bytes]

        # YOLO object det
        yolo_result = self.yolo_model.predict(images, imgsz=1600, conf=0.365, iou=0.1, max_det=10, verbose=False)  # max F1, try augment=True and adjusting iou
        yolo_result = [(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()) for r in yolo_result]
        yolo_result = [tuple(zip(*r)) for r in yolo_result]  # list of tuple[box, conf] in each image in xyxy format

        # crop the boxes out
        cropped_boxes = []
        for im, boxes in zip(images, yolo_result):
            im_boxes = []
            for (x1, y1, x2, y2), _ in boxes:
                im_boxes.append(im.crop((x1, y1, x2, y2)))
            cropped_boxes.append(im_boxes)

        captions_list = [[caption] for caption in captions]
        assert len(cropped_boxes) == len(captions_list)  # shld be == bs

        # clip inference
        clip_results = []
        with torch.cuda.amp.autocast():
            for boxes, im_captions in zip(cropped_boxes, captions_list):
                r = self.clip_model(boxes, candidate_labels=im_captions)
                # only 1 caption/img at test time so just use [0]
                image_to_text_scores = {im_captions[0]: [box[0]['score'] for box in r]}  # {caption: [score1, score2, ...]}, scores in sequence of bbox
                clip_results.append(image_to_text_scores)

        bboxes = []
        # combine the results
        for caption, yolo_box, similarity_scores in zip(captions, yolo_result, clip_results):
            box_idx = np.argmax(similarity_scores[caption])
            x1, y1, x2, y2 = yolo_box[box_idx][0]
            # convert to ltwh
            bboxes.append([x1, y1, x2 - x1, y2 - y1])

        logging.info(f'Captions:\n{captions}\nBoxes:\n{bboxes}')

        return bboxes


if __name__ == "__main__":
    from tqdm import tqdm
    import orjson
    import base64

    vlm_manager = VLMManager(yolo_path='yolov9c_0.99_0.769.pt', clip_path='CLIP-ViT-H-14-laion2B-s32B-b79K')
    all_answers = []

    batch_size = 4
    instances = []
    truths = []
    counter = 0
    max_samples = 12  # try on 12 samples only
    with open("../../data/vlm.jsonl", "r") as f:
        for line in tqdm(f):
            if counter > max_samples:
                break
            if line.strip() == "":
                continue
            instance = orjson.loads(line.strip())
            with open(f'../../data/images/{instance["image"]}', "rb") as file:
                image_bytes = file.read()
                for annotation in instance["annotations"]:
                    instances.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "b64": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    )
                    truths.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "bbox": annotation["bbox"],
                        }
                    )
                    counter += 1

    assert len(truths) == len(instances)

    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index: index + batch_size]
        input_data = {
            "instances": [
                {field: _instance[field] for field in ("key", "caption", "b64")}
                for _instance in _instances
            ]
        }
        img_bytes = [base64.b64decode(instance["b64"]) for instance in input_data["instances"]]
        captions = [instance["caption"] for instance in input_data["instances"]]
        bbox = vlm_manager.identify(img_bytes, captions)
        all_answers.extend(bbox)

    with open("results.json", "w") as f:
        orjson.dump(all_answers, f)
