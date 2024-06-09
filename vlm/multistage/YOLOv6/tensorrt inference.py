import numpy as np
import torch
from PIL import Image

from yolov6.data.data_augment import letterbox
from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression


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
    """Process image before image inference."""
    img_src = np.asarray(img)
    image = letterbox(img_src, img_size, stride=stride)[0]

    # Convert
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src


device = torch.device('cuda')

image = Image.open('../../../data/images/image_1.jpg')
img_size = check_img_size([870, 1520], s=32)
image, img_src = process_image(img=image, img_size=img_size, stride=32, half=True)
image = image.unsqueeze(0).to(device)

curr_model = DetectBackend('29_ckpt_yolov6l6_blind.pt', device=device)
model = curr_model.model.half().eval()

dummy = torch.ones(1, 3, 896, 1536, dtype=torch.float16, device=device)
model_trt = torch2trt(model, [dummy], output_names=['outputs'], fp16_mode=True, min_shapes=[(1, 3, 896, 1536)], opt_shapes=[(1, 3, 896, 1536)], max_shapes=[(1, 3, 896, 1536)], use_onnx=True)
torch.save(model_trt.state_dict(), '29_ckpt_yolov6l6_blind_trt.pth')

# model_trt = TRTModule()
# model_trt.load_state_dict(torch.load('29_ckpt_yolov6l6_blind_trt.pth'))
model_trt.output_flattener._schema = model_trt.output_flattener._schema[:1]  # fix output schema as we only need the first output
y = model(image)[0]
y_trt = model_trt(image)[0]
print('Vision model exported. atol:', torch.max(torch.abs(y - y_trt)))  # ~16

classes = None  # the classes to keep
nms_conf_thres: float = 0.01

# CHANGE THIS LINE FOR IOU THRESHOLD
iou_thres: float = 0.5
max_det: int = 10
agnostic_nms: bool = False

det = non_max_suppression(y_trt.unsqueeze(0), nms_conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
print(det)
