from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import time

model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    # model_path='../yolov9e_0.995_0.823_epoch65.pt',
    model_path='../yolov9c_0.99_0.769.pt',
    confidence_threshold=0.5,
    image_size=640,
    device="cuda",
)

from PIL import Image, ImageDraw
import numpy as np

timings = []
for i in range(10):
    im1 = Image.open(f'../../../data/images/image_{i}.jpg')
    s = time.perf_counter()
    result = get_sliced_prediction(im1, model, perform_standard_pred=False, postprocess_class_agnostic=True, batch=6, verbose=0).object_prediction_list
    timings.append(time.perf_counter() - s)
print(timings)
print(np.mean(timings[1:]))
result_n = [([r.bbox.minx / 1520, r.bbox.miny / 870, r.bbox.maxx / 1520, r.bbox.maxy / 870], r.score.value) for r in result]
result = [([r.bbox.minx, r.bbox.miny, r.bbox.maxx, r.bbox.maxy], r.score.value) for r in result]


for im, boxes in zip([im1], [result]):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), conf in boxes:
        draw.rectangle(xy=((x1, y1), (x2, y2)), outline='red')
        draw.text((x1, y1), text=f'{conf:.2f}', fill='red')
    im.show()