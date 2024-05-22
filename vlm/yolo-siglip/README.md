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

```

## TODO
1. Train YOLOv9c with 1600 resolution (now is 640 but infer at 1600 still helps)
2. Train YOLOv9c with noise augmentations
3. Add Real-ESRGAN to upscale the image before feeding to SigLIP
4. Add SAHI https://github.com/obss/sahi slicing inference for YOLO to detect small objects