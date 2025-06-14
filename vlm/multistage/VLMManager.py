import io
import logging
import math
import os
import sys

# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

sys.path.insert(0, "/workspace")
sys.path.insert(0, "yolov6/")
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer

from typing import List, Optional
import numpy as np
import torch
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from transformers import AutoImageProcessor, AutoModelForZeroShotImageClassification, AutoTokenizer, ZeroShotImageClassificationPipeline, SiglipTokenizer, SiglipImageProcessor
from torch2trt import TRTModule
from transformers.image_utils import load_image

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()]
)


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
        pass
        # print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size, list) else [new_size] * 2


def process_image(img, img_size, stride, half):
    """Process image before image inference."""
    img_src = np.asarray(img)
    image = letterbox(img_src, img_size, stride=stride)[0]

    # Convert
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src


class VLMManager:
    def __init__(self, yolo_paths: list[str], sliced_yolo_paths: list[str], clip_path: str, upscaler_path: str, use_sahi: list, siglip_trt: bool = False):
        logging.info(f'Loading {len(yolo_paths)} YOLO models from {yolo_paths}. Using SAHI: {use_sahi}. Using SigLIP TensorRT: {siglip_trt}')
        self.device = torch.device('cuda:0')

        self.use_sahi = use_sahi
        self.siglip_trt = siglip_trt

        assert len(self.use_sahi) == len(yolo_paths) == len(sliced_yolo_paths) == 1, \
            "WBF removed. Only 1 model can be used."

        self.isyolov6 = ['yolov6' in yolo_path for yolo_path in yolo_paths]
        self.sliced_isyolov6 = ['yolov6' in sliced_yolo_path for sliced_yolo_path in sliced_yolo_paths]
        assert self.isyolov6 == self.sliced_isyolov6, \
            "Sliced and non-sliced model must either be both YOLOv6 or non-YOLOv6"

        self.isyolov6_trt = [isyolov6 and yolo_path.endswith('.pth') for isyolov6, yolo_path in zip(self.isyolov6, yolo_paths)]

        self.yolo_models = []
        for yolo_path, sahi_yolo_path, use_sahi, isyolov6, isyolov6_trt in zip(yolo_paths, sliced_yolo_paths, self.use_sahi, self.isyolov6, self.isyolov6_trt):
            # pytorch or TRT (yolov6 TRT dont support SAHI)
            if isyolov6_trt:
                curr_model = TRTModule()
                curr_model.load_state_dict(torch.load(yolo_path))
                curr_model.output_flattener._schema = curr_model.output_flattener._schema[:1]  # fix output schema as we only need the first output
            else:
                curr_model = DetectBackend(yolo_path, device=self.device)

            self.yolo_models.append(curr_model)

        # self.yolo_wbf_weights = [1] * len(self.yolo_models)
        # assert len(self.yolo_models) == len(self.yolo_wbf_weights)

        logging.info(f'Warming up YOLO')
        for i in range(3):
            for use_sahi, is_yolov6, yolo_model, isyolov6_trt in zip(self.use_sahi, self.isyolov6, self.yolo_models, self.isyolov6_trt):
                if isyolov6_trt:
                    warmup_img_size = check_img_size([870, 1520], s=32)
                    yolo_model(torch.zeros((1, 3, *warmup_img_size), dtype=torch.float16, device=self.device))[0]  # noqa, warmup
                else:
                    warmup_img_size = check_img_size([870, 1520], s=yolo_model.stride)
                    yolo_model.model.half()
                    yolo_model(torch.zeros((1, 3, *warmup_img_size), dtype=next(yolo_model.model.parameters()).dtype, device=self.device))  # warmup

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
        if self.siglip_trt:
            self.clip_image_processor = SiglipImageProcessor.from_pretrained(clip_path)
            self.clip_tokenizer = SiglipTokenizer.from_pretrained(clip_path)
            self.clip_logit_scale_exp = torch.tensor([118.3125], device=self.device, dtype=torch.float16, requires_grad=False)
            self.clip_logit_bias = torch.tensor([-12.6640625], device=self.device, dtype=torch.float16, requires_grad=False)
            self.clip_vision_trt = TRTModule()
            self.clip_vision_trt.load_state_dict(torch.load(os.path.join(clip_path, 'vision_trt.pth')))
            self.clip_text_trt = TRTModule()
            self.clip_text_trt.load_state_dict(torch.load(os.path.join(clip_path, 'text_trt.pth')))

            logging.info(f'Warming up CLIP')
            for i in range(3):
                self.clip_vision_trt(torch.ones(20, 3, 384, 384, device=self.device, dtype=torch.float16))
                self.clip_text_trt(torch.ones(1, 64, device=self.device, dtype=torch.int64))
        else:
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
        torch.cuda.empty_cache()  # clear up vram for inference

        # image is the raw bytes of a JPEG file
        images = [Image.open(io.BytesIO(b)) for b in img_bytes]
        yolo_results = []
        # YOLO object det with WBF
        for use_sahi, is_yolov6, yolo_model, isyolov6_trt in zip(self.use_sahi, self.isyolov6, self.yolo_models, self.isyolov6_trt):
            img_size = check_img_size([870, 1520], s=yolo_model.stride if not isyolov6_trt else 32)
            yolo_result = []
            for image in images:
                img, img_src = process_image(img=image, img_size=img_size, stride=yolo_model.stride if not isyolov6_trt else 32, half=True)
                img = img.unsqueeze(0).to(self.device)  # expand batch dim
                pred_results = yolo_model(img)
                if isyolov6_trt: pred_results = pred_results[0].unsqueeze(0)  # TRT outputs 2 but 2nd is useless, need expand trt batch dim too since torch2trt flattens it

                classes: Optional[List[int]] = None  # the classes to keep
                nms_conf_thres: float = 0.01

                # CHANGE THIS LINE FOR IOU THRESHOLD
                iou_thres: float = 0.3
                max_det: int = 20
                agnostic_nms: bool = False

                det = non_max_suppression(pred_results, nms_conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

                # CHANGE THIS LINE FOR CONFIDENCE THRESHOLD
                filter_conf_thres = 0.5
                curr_img_detections = []
                if len(det):
                    det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape)
                    # Filter out the detections based on the confidence threshold
                    filtered_det = det[det[:, 4] >= filter_conf_thres]

                    # If there are no detections that pass the threshold, use the one with the highest confidence
                    if len(filtered_det) == 0:
                        filtered_det = det[-1].unsqueeze(0)

                    # Normalize the bounding box coordinates
                    # norm_tensor = torch.tensor([1520, 870, 1520, 870], device=self.device)
                    normalized_xyxy = filtered_det[:, :4]

                    # Combine the normalized coordinates and the confidence scores
                    curr_img_detections = torch.cat((normalized_xyxy, filtered_det[:, 4].unsqueeze(1)), dim=1).tolist()
                    # curr_img_detections = torch.ones((0,)).tolist()

                    # if not curr_img_detections:
                    #     # nothing passes filter_conf_thres so just use the highest conf pred
                    #     # det[-1, :4] /= norm_tensor
                    #     curr_img_detections = [det[-1, :5].tolist()]

                    # print("Curr detections:", curr_img_detections)

                yolo_result.append(curr_img_detections)
            yolo_results.append(yolo_result)

        # crop the boxes out
        cropped_boxes = []
        for im, boxes in zip(images, yolo_results[0]):
            im_boxes = []
            for *xyxy, conf in boxes:
                cropped = im.crop(xyxy)
                cropped = np.asarray(cropped)
                if not any(s <= 10 for s in cropped.shape[:2]):
                    cropped = self.upscaler_pad10.enhance(cropped, outscale=4)[0]
                elif not any(s <= 1 for s in cropped.shape[:2]):
                    cropped = self.upscaler_pad1.enhance(cropped, outscale=4)[0]
                else:  # upscaler transposes CHW to HWC, so if upscaler not called then transpose manually
                    cropped = cropped.astype(np.uint8)
                cropped = Image.fromarray(cropped)
                im_boxes.append(cropped)
            cropped_boxes.append(im_boxes)

        captions_list = [[caption] for caption in captions]
        assert len(cropped_boxes) == len(captions_list)  # shld be == bs

        # clip inference
        clip_results = []
        with torch.no_grad():
            for boxes, im_captions in zip(cropped_boxes, captions_list):
                boxes = [img.resize((dim + 1 for dim in img.size), resample=Image.LANCZOS) if any(i == 1 for i in img.size) else img for img in boxes]  # fix bug in HF preprocessing where it wrongly takes any dim with 1 as channel(grayscale) when it could be h/w
                # print("Boxes:", boxes)
                if self.siglip_trt:
                    im_captions_templated = [f'This is a photo of {caption}.' for caption in im_captions]  # prompt template used in HF pipeline
                    vision_input = self.clip_image_processor(images=boxes, return_tensors='pt').to(self.device)
                    text_inputs = self.clip_tokenizer(im_captions_templated, return_tensors='pt', padding='max_length', truncation=True).to(self.device)  # processor wont work since it dont pad to max_length=64
                    vision_input = vision_input['pixel_values'].type(torch.float16)
                    image_feat = self.clip_vision_trt(vision_input)['pooler_output']
                    text_feat = self.clip_text_trt(text_inputs['input_ids'])['pooler_output']
                    image_feat /= image_feat.norm(p=2, dim=-1, keepdim=True)
                    text_feat /= text_feat.norm(p=2, dim=-1, keepdim=True)
                    scores = image_feat @ text_feat.T * self.clip_logit_scale_exp + self.clip_logit_bias
                    scores = scores.squeeze(-1).tolist()  # sigmoid not needed as it dont change the ranking
                    if not isinstance(scores, list): scores = [scores]
                    im_captions *= len(scores)  # repeat the captions to match the scores for zip() below
                    r = [{"score": score, "label": candidate_label} for score, candidate_label in zip(scores, im_captions)]  # cannot sort here else scramble the box idx
                    # only 1 caption at test time so just use [0]
                    image_to_text_scores = {im_captions[0]: [box['score'] for box in r]}  # {caption: [score1, score2, ...]}, score in seq of bbox
                else:
                    r = self.clip_model(boxes, candidate_labels=im_captions)
                    # only 1 caption/img at test time so just use [0]
                    image_to_text_scores = {im_captions[0]: [box[0]['score'] for box in r]}  # {caption: [score1, score2, ...]}, scores in sequence of bbox
                clip_results.append(image_to_text_scores)

        bboxes = []
        # combine the results
        for caption, yolo_box, similarity_scores in zip(captions, yolo_results[0], clip_results):
            try:
                box_idx = np.argmax(similarity_scores[caption])
                x1, y1, x2, y2, conf = yolo_box[box_idx]
                # convert to ltwh
                bboxes.append([x1, y1, x2 - x1, y2 - y1])
            except Exception:
                bboxes.append([300, 300, 50, 50])

        logging.info(f'Captions:\n{captions}\nBoxes:\n{bboxes}')

        return bboxes


if __name__ == "__main__":
    from tqdm import tqdm
    import orjson
    import base64

    vlm_manager = VLMManager(yolo_paths=['YOLOv6/29_ckpt_yolov6l6_blind_trt.pth'], sliced_yolo_paths=['YOLOv6/29_ckpt_yolov6l6_blind_trt.pth'], clip_path='siglip/siglip-large-epoch5-augv2-upscale_0.892_cont_5ep_0.905', upscaler_path='realesr-general-x4v3.pth', use_sahi=[False])
    all_answers = []

    batch_size = 1
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

    with open("results.json", "wb") as f:
        f.write(orjson.dumps(all_answers))
