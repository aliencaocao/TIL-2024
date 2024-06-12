import torch
from torch2trt import torch2trt
from transformers import SiglipModel

model = SiglipModel.from_pretrained('siglip-large-epoch5-augv2-upscale_0.892_cont_5ep_0.905', torch_dtype=torch.float16).cuda().eval()

text_model = model.text_model
vision_model = model.vision_model

dummy = torch.ones(1, 3, 384, 384, dtype=torch.float16, device='cuda')
model_trt = torch2trt(vision_model, [dummy], fp16_mode=True, min_shapes=[(1, 3, 384, 384)], opt_shapes=[(10, 3, 384, 384)], max_shapes=[(20, 3, 384, 384)], use_onnx=True)
y = vision_model(dummy).pooler_output
y_trt = model_trt(dummy)['pooler_output']
torch.save(model_trt.state_dict(), 'vision_trt.pth')
print('Vision model exported. atol:', torch.max(torch.abs(y - y_trt)))

dummy = torch.ones(1, 64, dtype=torch.long, device='cuda')
model_trt = torch2trt(text_model, [dummy], fp16_mode=True, min_shapes=[(1, 64)], opt_shapes=[(1, 64)], max_shapes=[(1, 64)], use_onnx=True)
y = text_model(dummy).pooler_output
y_trt = model_trt(dummy)['pooler_output']
torch.save(model_trt.state_dict(), 'text_trt.pth')
print('Text model exported. atol:', torch.max(torch.abs(y - y_trt)))
