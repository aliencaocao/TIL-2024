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
2. SigLIP-SO400M-patch14-384
```shell
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384 --local-dir-use-symlinks False
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
gcloud ai models upload --region asia-southeast1 --display-name '12000sgd-multistage-vlm' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:latest --container-health-route /health --container-predict-route /extract --container-ports 5004 --version-aliases default
```

### Evaluation
#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + SigLIP
val set 0.7987355110642782

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

Conclusion: pre_pad 1 or 10 dont make much diff, but speed increase VS acc improvement is good. Still worse than without upscaling though.

**ESRGAN not worth it. Does not improve accuracy.**

## TODO
1. Train YOLOv9c with 1600 resolution (now is 640 but infer at 1600 still helps)
2. Train YOLOv9c with noise augmentations
3. Manual impl slicing inference (batched) for YOLO to detect small objects, tried yolo-patched-inference and it sucks