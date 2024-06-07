# import os
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

import gc
import io
import logging
import math

import sys

sys.path.insert(0, ".")
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer

from typing import List, Optional
import numpy as np
import torch
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from transformers import AutoImageProcessor, AutoModelForZeroShotImageClassification, AutoTokenizer, ZeroShotImageClassificationPipeline
from transformers.image_utils import load_image
from ultralytics import YOLO

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler()
    ])


class CustomPipeline(ZeroShotImageClassificationPipeline):
    def preprocess(self, image, candidate_labels=None, hypothesis_template="This is a photo of {}.", timeout=None):
        image = load_image(image, timeout=timeout)
        inputs = self.image_processor(images=[image], return_tensors=self.framework)
        inputs["pixel_values"] = inputs["pixel_values"].type(self.torch_dtype)  # cast to whatever dtype model is in (previously always in fp32)
        inputs["candidate_labels"] = candidate_labels
        sequences = [hypothesis_template.format(x) for x in candidate_labels]
        padding = "max_length" if self.model.config.model_type == "siglip" else True
        text_inputs = self.tokenizer(sequences, return_tensors=self.framework, padding=padding)
        inputs["text_inputs"] = [text_inputs]
        return inputs

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


def check_img_size(img_size, s=32, floor=0):
    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size, list) else [new_size] * 2


def process_image(img, img_size, stride, half):
    '''Process image before image inference.'''
    img_src = np.asarray(img)
    image = letterbox(img_src, img_size, stride=stride)[0]

    # Convert
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src


class VLMManager:
    def __init__(self, yolo_paths: list[str], clip_path: str, upscaler_path: str, use_sahi: bool = True):
        logging.info(f'Loading {len(yolo_paths)} YOLO models from {yolo_paths}. Using SAHI: {use_sahi}')
        self.device = torch.device('cuda:0')

        self.use_sahi = use_sahi
        if self.use_sahi:
            self.yolo_models = [AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=yolo_path,
                confidence_threshold=0.5,
                image_size=896,  # not used for TRT. TRT uses cfg below
                standard_pred_image_size=1600,  # not used for TRT. TRT uses cfg below
                device="cuda",
                cfg={"task": 'detect', "names": {'0': 'target'}, "standard_pred_image_size": (960, 1600), "standard_pred_model_path": f'{yolo_path.rsplit(".")[0]}_bs1.engine', "imgsz": (768, 896), "half": True}
            ) for yolo_path in yolo_paths]
        else:
            self.yolo_models = [YOLO(yolo_path) if "yolov6" not in yolo_path else DetectBackend(yolo_path, device=self.device) for yolo_path in yolo_paths]

        self.isyolov6 = ['yolov6' in yolo_path for yolo_path in yolo_paths]
        for i, isyolov6 in enumerate(self.isyolov6):
            if isyolov6:
                self.yolo_models[i].model.eval().half()

        self.yolo_wbf_weights = [1] * len(self.yolo_models)
        assert len(self.yolo_models) == len(self.yolo_wbf_weights)

        logging.info(f'Warming up YOLO')
        for i in range(3):
            ctr = 0
            for yolo_model in self.yolo_models:
                if self.use_sahi:
                    get_sliced_prediction(Image.new('RGB', (1520, 870)), yolo_model, perform_standard_pred=True, postprocess_class_agnostic=True, batch=6, verbose=0).object_prediction_list  # noqa
                elif self.isyolov6[ctr]:
                    warmup_img_size = check_img_size([1520, 870], s=yolo_model.stride)
                    yolo_model(torch.zeros(1, 3, *warmup_img_size).to(self.device).type_as(next(yolo_model.model.parameters())))  # warmup
                else:
                    yolo_model.predict(Image.new('RGB', (1520, 870)), imgsz=1600, conf=0.5, iou=0.1, max_det=10, verbose=False, augment=True)  # warmup
                ctr += 1
        logging.info(f'Loading upscaler model from {upscaler_path}')
        rrdb_net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        self.upscaler_pad10 = RealESRGANer(
            scale=4,
            model_path=upscaler_path,
            model=rrdb_net,
            pre_pad=10,
            half=True)
        self.upscaler_pad1 = RealESRGANer(
            scale=4,
            model_path=upscaler_path,
            model=rrdb_net,
            pre_pad=1,
            half=True)
        logging.info(f'Warming up upscaler')
        for i in range(3):
            self.upscaler_pad10.enhance(np.zeros((50, 50, 3), dtype=np.uint8), outscale=4)  # warmup
            self.upscaler_pad1.enhance(np.zeros((8, 10, 3), dtype=np.uint8), outscale=4)  # warmup

        logging.info(f'Loading CLIP model from {clip_path}')
        self.clip_model = CustomPipeline(task="zero-shot-image-classification",
                                         model=AutoModelForZeroShotImageClassification.from_pretrained(clip_path, torch_dtype=torch.float16),
                                         tokenizer=AutoTokenizer.from_pretrained(clip_path),
                                         image_processor=AutoImageProcessor.from_pretrained(clip_path),
                                         batch_size=4, device='cuda')
        logging.info(f'Warming up CLIP')
        for i in range(3):
            self.clip_model(images=Image.new('RGB', (50, 50)), candidate_labels=['hello'])  # warmup
        logging.info('VLMManager initialized')

    def identify(self, img_bytes: list[bytes], captions: list[str]) -> list[list[int]]:
        logging.info('Predicting')
        gc.collect()
        torch.cuda.empty_cache()  # clear up vram for inference

        # image is the raw bytes of a JPEG file
        images = [Image.open(io.BytesIO(b)) for b in img_bytes]
        yolo_results = []
        # YOLO object det with WBF
        ctr = 0
        for yolo_model in self.yolo_models:
            if self.use_sahi:
                yolo_result = []
                for image in images:
                    per_img_result = get_sliced_prediction(image, yolo_model, perform_standard_pred=True, postprocess_class_agnostic=True, batch=6, verbose=0).object_prediction_list
                    per_img_result = [([r.bbox.minx / 1520, r.bbox.miny / 870, r.bbox.maxx / 1520, r.bbox.maxy / 870], r.score.value) for r in per_img_result]
                    yolo_result.append(per_img_result)
            elif self.isyolov6[ctr]:
                img_size = check_img_size([1520, 870], s=yolo_model.stride)
                yolo_result = []
                for image in images:
                    img, img_src = process_image(img=image, img_size=img_size, stride=yolo_model.stride, half=True)
                    img = img.to(self.device)
                    if len(img.shape) == 3:
                        img = img[None]
                        # expand for batch dim
                    pred_results = yolo_model(img)
                    classes: Optional[List[int]] = None  # the classes to keep
                    conf_thres: float = 0.25
                    iou_thres: float = 0.3
                    max_det: int = 10
                    agnostic_nms: bool = False
                    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
                    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    img_ori = img_src.copy()
                    if len(det):
                        det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                        curr_img_detections = []
                        for *xyxy, conf, cls in reversed(det):
                            xyxy = [xyxy[0] / 1520, xyxy[1] / 870, xyxy[2] / 1520, xyxy[3] / 870]
                            curr_img_detections.append([[x.item() for x in xyxy], conf.item()])
                    yolo_result.append(curr_img_detections)
            else:
                yolo_result = yolo_model.predict(images, imgsz=1600, conf=0.5, iou=0.1, max_det=10, verbose=False, augment=True)
                yolo_result = [(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist()) for r in yolo_result]  # WBF need normalized xyxy
                yolo_result = [tuple(zip(*r)) for r in yolo_result]  # list of tuple[box, conf] in each image
                print(yolo_result)
            yolo_results.append(yolo_result)

        wbf_boxes = []
        for i, img in enumerate(images):
            boxes_list = []
            scores_list = []
            labels_list = []
            for yolo_result in yolo_results:
                boxes_list.append([r[0] for r in yolo_result[i]])
                scores_list.append([r[1] for r in yolo_result[i]])
                labels_list.append([0] * len(yolo_result[i]))
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=self.yolo_wbf_weights, iou_thr=0.5, skip_box_thr=0.0001)
            boxes = boxes.tolist()
            # normalize
            w, h = img.size
            boxes = [[x1 * w, y1 * h, x2 * w, y2 * h] for x1, y1, x2, y2 in boxes]
            wbf_boxes.append(boxes)

        assert len(wbf_boxes) == len(images)  # shld be == bs

        # crop the boxes out
        cropped_boxes = []
        for im, boxes in zip(images, wbf_boxes):
            im_boxes = []
            for x1, y1, x2, y2 in boxes:
                cropped = im.crop((x1, y1, x2, y2))
                cropped = np.asarray(cropped)
                if not any(s <= 10 for s in cropped.shape[:2]):
                    cropped = self.upscaler_pad10.enhance(cropped, outscale=4)[0]
                else:
                    cropped = self.upscaler_pad1.enhance(cropped, outscale=4)[0]
                cropped = Image.fromarray(cropped)
                im_boxes.append(cropped)
            cropped_boxes.append(im_boxes)

        captions_list = [[caption] for caption in captions]
        assert len(cropped_boxes) == len(captions_list)  # shld be == bs

        # clip inference
        clip_results = []
        with torch.no_grad():
            for boxes, im_captions in zip(cropped_boxes, captions_list):
                r = self.clip_model(boxes, candidate_labels=im_captions)
                # only 1 caption/img at test time so just use [0]
                image_to_text_scores = {im_captions[0]: [box[0]['score'] for box in r]}  # {caption: [score1, score2, ...]}, scores in sequence of bbox
                clip_results.append(image_to_text_scores)

        bboxes = []
        # combine the results
        try:
            for caption, yolo_box, similarity_scores in zip(captions, wbf_boxes, clip_results):
                box_idx = np.argmax(similarity_scores[caption])
                x1, y1, x2, y2 = yolo_box[box_idx]
                # convert to ltwh
                bboxes.append([x1, y1, x2 - x1, y2 - y1])
        except:
            bboxes = []

        logging.info(f'Captions:\n{captions}\nBoxes:\n{bboxes}')

        return bboxes


if __name__ == "__main__":
    from tqdm import tqdm
    import orjson
    import base64

    vlm_manager = VLMManager(yolo_paths=['best_yolov6l6.pt'], clip_path='siglip-large-patch16-384-ft', upscaler_path='realesr-general-x4v3.pth', use_sahi=False)
    all_answers = []

    batch_size = 4
    instances = []
    truths = []
    counter = 0
    max_samples = 12  # try on 12 samples only
    with open("../../../advanced/vlm.jsonl", "r") as f:
        for line in tqdm(f):
            if counter > max_samples:
                break
            if line.strip() == "":
                continue
            instance = orjson.loads(line.strip())
            with open(f'../../../advanced/images/{instance["image"]}', "rb") as file:
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

    with open("results.json", "wb") as f:
        f.write(orjson.dumps(all_answers))
