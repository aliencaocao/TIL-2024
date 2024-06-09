import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

device = torch.device('cuda')

logger = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(logger, namespace="")
with open('29_ckpt_yolov6l6_blind.engine', 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())
context = model.create_execution_context()


def letterbox(im, new_shape=(896, 1536), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def postprocess(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes.clip_(0,6400)


origin_RGB = []
resize_data = []
img = cv2.imread('../../../data/images/image_1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
origin_RGB.append(img)
image = img.copy()
image, ratio, dwdh = letterbox(image, auto=False)
image = image.transpose((2, 0, 1))
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)
im = image.astype(np.float16)
resize_data.append((im,ratio,dwdh))


def getBindings(model, context, shape=(1,3,896,1536)):
    context.set_input_shape('images', shape)
    bindings = OrderedDict()
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

    for index in range(model.num_io_tensors):
        name = model.get_tensor_name(index)
        shape = tuple(context.get_tensor_shape(name))
        dtype = np.float16
        data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    return bindings


# warmup for 10 times
bindings = getBindings(model,context,(1,3,896,1536))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
for _ in range(10):
    tmp = torch.randn(1,3,896,1536).to(device)
    binding_addrs['images'] = int(tmp.data_ptr())
    context.execute_v2(list(binding_addrs.values()))

np_batch = np.concatenate([data[0] for data in resize_data])
print(np_batch.shape)

batch_1 = torch.from_numpy(np_batch[0:1]).to(device)/255
bindings = getBindings(model,context,(1,3,640,640))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

start = time.perf_counter()
binding_addrs['images'] = int(batch_1.data_ptr())
context.execute_v2(list(binding_addrs.values()))
print(f'Cost {time.perf_counter()-start} s')


nums = bindings['num_dets'].data
boxes = bindings['det_boxes'].data
scores = bindings['det_scores'].data
classes = bindings['det_classes'].data
print(nums.shape,boxes.shape,scores.shape,classes.shape)

for batch,(num,box,score,cls) in enumerate(zip(nums.flatten(),boxes,scores,classes)):
    if batch>5:
        break
    RGB = origin_RGB[batch]
    ratio,dwdh = resize_data[batch][1:]
    box = postprocess(box.clone(),ratio,dwdh).round().int()
    for idx,(b,s,c) in enumerate(zip(box,score,cls)):
        b,s,c = b.tolist(),round(float(s),3),int(c)
        cv2.rectangle(RGB,b[:2],b[2:],(255, 0, 0),2)
        cv2.putText(RGB,'object' + ' ' + str(s),(b[0], b[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255, 0, 0),thickness=2)

Image.fromarray(origin_RGB[0]).show()

