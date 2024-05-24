# Muli Stage approach
## Overview
1. YOLOv9c trained on single-class detection of targets in general
2. Extract the bboxes as deteced by YOLO
3. Feed each bbox into SigLIP-SO400M-patch14-384 pretrained by Google and get similarity score VS caption (1/image)
4. Choose the box with the highest similarity score

## Models
1. YOLOv9c
https://drive.google.com/file/d/1EpI4Y7cbZrtwYEnEhKmdwrvDkFOcm-jZ
```shell
gdown 1EpI4Y7cbZrtwYEnEhKmdwrvDkFOcm-jZ
```
2. CLIP of choice
```shell
huggingface-cli download facebook/metaclip-b16-fullcc2.5b --local-dir metaclip-b16-fullcc2.5b --local-dir-use-symlinks False --exclude *.bin
```

### Docker
```shell
docker build -t 12000sgd-multistage-vlm .
```
Test:
```shell
docker run -p 5004:5004 --gpus all -d 12000sgd-multistage-vlm
```
Submit:
```shell
docker tag 12000sgd-multistage-vlm asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:latest
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:latest
gcloud ai models upload --region asia-southeast1 --display-name '12000sgd-multistage-vlm' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:latest --container-health-route /health --container-predict-route /identify --container-ports 5004 --version-aliases default
```

### Evaluation
#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + SigLIP
val set 0.7987355110642782

test set:
- Accuracy: 0.667
- Speed Score: 0.7041770444444444

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + openai/clip-vit-large-patch14
val set 0.8203371970495258

not testing due to low perf on val

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + openai/clip-vit-large-patch14-336
val set 0.8159466104671584

not testing due to low perf on val

high res degrade perf


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + CLIP-ViT-H-14-laion2B-s32B-b79K
val set 0.8767123287671232

with upscale x2 pad=10: val set 0.7600260841212911

with upscale x4v3 pad=1: val set 0.41789445486204124

with upscale x4v3 pad=10: val set 0.819634703196347

test set: 

Upscaling still bad

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + EVA02-CLIP-L-14
30% faster than any other VIT-L models, maybe a HF pipeline overhead issue

val set fp16 0.6457674745345978

val set fp32 AMP 0.6454162276080084

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + EVA02-CLIP-L-14-336
A bit slower than 224 but still 20% faster than other VIT-L

val set 0.6434843695117668

Conclusion: EVA02 CLIP sucks

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + yolo-metaclip-b16-400m
val set 0.7799438004917457

not testing as clearly worse

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-b16-fullcc2.5b
val set 0.850895679662803

test set:

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-b32-400m
val set 0.7695820161573587

not testing as clearly bad


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-b32-fullcc2.5b
val set 0.847383210396909

not testing

Conclusion: b16 better than b32 marginally on val set


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-l14-400m
val set 0.8205128205128205

not testing as clearly worse than fullcc2.5b

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-l14-fullcc2.5b
val set 0.8640674394099052

with upscale x4v3 pad=10: val set 0.815595363540569

test set:


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-l14-400m
val set 0.8205128205128205

not testing


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-h14-fullcc2.5b
val set 0.8739023533544081

with upscale x4v3 pad=10: val set 0.8243765367053039

test set:

From val set only, H series has the best perf compared to L, L is better than B. More data lead to better.


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + ESRGANx2 + SigLIP
pre_pad=1: val set 0.672641652741911

pre_pad=10: val set 0.7766069546891464

Conclusion: ESRGANx2 is bad at prepad=1


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + ESRGANx4 + SigLIP
pre_pad=1: val set 0.7809975412715139

pre_pad=10: val set 0.720480753613773

Conclusion: ESRGANx4 is a lot better than 2x but 20% slower at prepad=1, ESRGANx4 is bad at prepad=10 somehow, but overall 4x is better than 2x


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + ESRGANx4v3 + SigLIP
x4v3 is the lite model, 100% faster!

pre_pad=1: val set 0.782051282051282

pre_pad=10: val set 0.7804706708816298

Conclusion: pre_pad 1 or 10 dont make much diff, but speed increase VS acc improvement is good. Still worse than without upscaling though. realesr-general-x4v3 is better than normal x4 marginally.

**ESRGAN not worth it. Does not improve accuracy on all CLIPs.**


## TODO
1. Train YOLOv9c with 1600 resolution (now is 640 but infer at 1600 still helps)
2. Train YOLOv9c with noise augmentations
3. Manual impl slicing inference (batched) for YOLO to detect small objects, tried yolo-patched-inference and it sucks
4. Try speed diff of pipeline VS openclip
5. Eval YOLOv9c with augment=true on inference
