from typing import List
from PIL import Image
import numpy as np
import io
from time import time

from transformers import pipeline

import os
import sys
sys.path.insert(0, "/codet_main/third_party/CenterNet2")
sys.path.insert(0, ".")
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from centernet.config import add_centernet_config
from codet.config import add_codet_config
from codet.modeling.utils import reset_cls_infer
from codet.modeling.text.text_encoder import build_text_encoder
from detectron2.engine.defaults import DefaultPredictor

class Args:
    def __init__(self, config_file, webcam, cpu, video_input, output, pred_all_class, confidence_threshold, opts):
        self.config_file = config_file
        self.webcam = webcam
        self.cpu = cpu
        self.video_input = video_input
        self.output = output
        self.pred_all_class = pred_all_class
        self.confidence_threshold = confidence_threshold
        self.opts = opts


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_codet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg

def get_clip_embeddings(vocabulary, text_encoder, prompt='a '):
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

class VLMManager:
    def __init__(self):
        args = Args(
            config_file="configs/CoDet_OVLVIS_EVA_4x.yaml",
            webcam=None,
            cpu=False,
            video_input=None,
            output="output/",
            pred_all_class=False,
            confidence_threshold=0.1,
            opts=['MODEL.WEIGHTS', 'CoDet_OVLVIS_EVA_4x.pth'],
        )
        cfg = setup_cfg(args)
        self.predictor = DefaultPredictor(cfg)
        print(self.predictor)

        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()
        
        print("Model loaded.")

    def identify(self, img_bytes: bytes, caption: str) -> List[int]:
        # image is the raw bytes of a JPEG file
        img = Image.open(io.BytesIO(img_bytes))
        
        img_np = np.array(img)
        img_bgr_np = img_np[:, :, ::-1]
        #img_bgr = Image.fromarray(img_bgr_np)
        #New code starts here
        thing_classes = [caption]
        num_classes = len(thing_classes)
        classifier = get_clip_embeddings(thing_classes, self.text_encoder)
        reset_cls_infer(self.predictor.model, classifier, num_classes)
        predictions = self.predictor(img_bgr_np)
        instances = predictions["instances"].to('cpu')
        bbox, score, class_id = instances.pred_boxes.tensor.tolist(), instances.scores.tolist(), instances.pred_classes.tolist()

        # save the highest conf box for each class id only
        bboxes = [(b, s) for b, s in zip(bbox, score)]
        highest_conf_pred = ""
        if bboxes:
            highest_conf_pred = max(bboxes, key=lambda x: x[1])
            highest_conf_pred = (*highest_conf_pred, caption)
            
        if highest_conf_pred == "":
            return [0, 0, 0, 0]
        
        bbox = highest_conf_pred[0]
        return [
            bbox[0],
            bbox[1],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
        ]
#         draw = ImageDraw.Draw(img)
#         for pred in preds:
#             box = pred["box"]
#             label = pred["label"]
#             score = pred["score"]

#             if score < 0.3:
#                 continue

#             xmin, ymin, xmax, ymax = box.values()
#             draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
#             draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")
#         img.save(f"/home/jupyter/TIL-2024/owlv2-preds/{caption}_{int(time())}.png")
        
        