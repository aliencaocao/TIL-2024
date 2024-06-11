import sys
sys.path.insert(0, '../multistage/YOLOv6')

from sahi import AutoDetectionModel
from sahi.models.yolov6 import Yolov6DetectionModel
from sahi.predict import get_sliced_prediction

from PIL import Image, ImageDraw

image = Image.open('../multistage/image_484.jpg')
draw = ImageDraw.Draw(image)

curr_model = Yolov6DetectionModel(
                        model_path='../multistage/29_ckpt_yolov6l6_blind.pt',
                        device="cuda",
                        category_mapping={"0": "target"},
                        # SETTINGS HERE
                        nms_confidence_threshold=0.01,
                        iou_threshold=0.3,
                        filter_confidence_threshold=0.5,
                        # image_size=[896, 768],
                        half=False,  # doesnt work on GTX 1650 laptop
                    )


per_img_result = get_sliced_prediction(
                        image,
                        curr_model,
                        perform_standard_pred=True,
                        postprocess_class_agnostic=True,
                        batch=6,
                        verbose=2,
                    ).object_prediction_list

per_img_result = [([r.bbox.minx, r.bbox.miny, r.bbox.maxx, r.bbox.maxy], r.score.value) for r in per_img_result]
for xyxy, conf in per_img_result:
    draw.rectangle(xyxy, fill=None, outline="red")
    draw.text((xyxy[0], xyxy[1]), f'{conf:.2f}', fill="red")
image.show()
