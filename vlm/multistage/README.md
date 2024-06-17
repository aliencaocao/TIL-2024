# Muli Stage approach
## Overview
1. YOLOv6l6 trained on single-class detection of targets in general, using custom data too
2. Extract the bboxes as detected by YOLO, optionally using SAHI (Slicing Aided Hyper Inference), helped for v9e but not v6l6
3. Run each extracted bbox through Real-ESRGAN x4v3 model to upscale 4x
4. Feed each bbox into a SigLIP and get similarity score VS caption (1/image)
5. Choose the box with the highest similarity score for each caption

## Models
1. YOLOv9e
https://drive.google.com/file/d/13VUjoienvWqHz9NB66KHDqL5Y0t3YCMN/view?usp=sharing
```shell
gdown 13VUjoienvWqHz9NB66KHDqL5Y0t3YCMN
```
2. CLIP of choice (best so far is siglip-large-patch16-384)
```shell
huggingface-cli download google/siglip-large-patch16-384 --local-dir siglip-large-patch16-384 --local-dir-use-symlinks False
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
docker tag 12000sgd-multistage-vlm asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:yolo-siglip-large-patch16-384-conf0.1
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:yolo-siglip-large-patch16-384-conf0.1
gcloud ai models upload --region asia-southeast1 --display-name '12000sgd-multistage-yolov9e-last-siglip-large-patch16-384-conf0.1-vlm' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:yolo-siglip-large-patch16-384-conf0.1 --container-health-route /health --container-predict-route /identify --container-ports 5004 --version-aliases default
```
Finals submission:
```shell
docker tag 12000sgd-multistage-vlm asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgdplushie-vlm:finals
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgdplushie-vlm:finals
```

## VRAM
- all model loaded: 2.6G
- yolo pred single image = max 3.2G (first inference has spike of 4.8G)
- upscaler use less than 10mb
- siglip so400m = max 2.9g @ 10boxes
- Overall peak needed: 4.8G


### Training YOLO Augs V1
1. Initialize from YOLOv9e checkpoint
2. Train for 55 epochs with AdamW, lr=1e-3, effective bs=64, image size=1280, cosine LR schedule
3. Continue for 7 epochs with image size=1600 to improve on high-res and small object further since inference time we use 1600

YOLO Augs V1:
```python
T = [
    A.GaussNoise(var_limit=2500, p=0.5),
    A.Flip(p=0.5),
    A.Blur(p=0.1),
    A.MedianBlur(p=0.1),
    A.ToGray(p=0.1),
    A.CLAHE(p=0.1),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.2),
    A.ImageCompression(quality_lower=75, p=0.5),
]
```

YOLO Augs V2:
```python
T = [
    A.GaussNoise(var_limit=(500, 2500), p=1.0, per_channel=True),
    A.ISONoise(p=1.0, color_shift=(0.02, 0.07)),
    A.MultiplicativeNoise(p=1.0),
    A.AdvancedBlur(blur_limit=(3, 7), p=0.2),
    A.Flip(p=0.5),
    A.CLAHE(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),
    A.RandomGamma(p=0.2),
]
```

AugsV2 proven to be bad.

YOLO Augs V3:
```python
T = [
    A.GaussNoise(var_limit=2500, p=0.5),
    A.ISONoise(p=0.5),
    A.Flip(p=0.5),
    A.Blur(p=0.1),
    A.MedianBlur(p=0.1),
    A.ToGray(p=0.1),
    A.CLAHE(p=0.1),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.2),
    A.ImageCompression(quality_lower=75, p=0.5),
]
```

### Training SigLIP using HF
1. Copy modeling_siglip.py from https://github.com/huggingface/transformers/blob/bdb9106f247fca48a71eb384be25dbbd29b065a8/src/transformers/models/siglip/modeling_siglip.py
2. Add loss func adapted from JAX in https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/trainers/proj/image_text/siglip.py#L287 to https://github.com/huggingface/transformers/blob/bdb9106f247fca48a71eb384be25dbbd29b065a8/src/transformers/models/siglip/modeling_siglip.py#L1230
```python
eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
nll = -torch.sum(loglik, dim=-1)
loss = nll.mean()
```
3. Multi-GPU training using FDSP:
```shell
CUDA_VISIBLE_DEVICES=0,1,4,5,6 accelerate launch HF_train.py
```
4. Run convert_safetensors.py as models trained with torch FDSP + torch dynamo has a prefix in model dict key names


### ~~Training SigLIP using JAX~~ Did not get to work, using HF with custom loss instead
Generate TensorFlow datasets:

Run split ds notebook, then
```shell
cd vlm\multistage\siglip\big_vision\datasets\til
tfds build
```
Setup Env:
```shell
cd vlm/multistage/siglip
pip install "jax[tpu]>=0.4.25" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r big_vision/requirements.txt
```
Train:
```shell
TFDS_DATA_DIR=/kaggle/input/til-siglip-tfds BV_JAX_INIT=1 python3 -m big_vision.trainers.proj.image_text.siglip --config big_vision/configs/proj/image_text/siglip_til.py --workdir til_train
```


### Takeaways
- Inferencing on YOLO with high res (1600px) brings noticeable improvement even using weights trained on 640px. Further training on 1280px and then 1600px significantly improves. This is a clear characteristic of small object detection tasks.
- Val and test correlation on CLIPs are not reliable beyond 0.8 mAP due to lack of noisy val data
- Upscaling is always bad on val even though they do result in a clearer segregation of scores in CLIPs. They do improve test scores. This is likely due to local testing samples are not noisy enough and benefit of upscaling is overweighed by the artifacts.
- SAHI (slicing inference) on YOLO requires much higher confidence score (0.1 vs 0.5) to reduce FPs
- Strong (var=2500) GaussianNoise augmentations significantly improve test performance of YOLO
- Reason why CLIP-ViT-H and Metaclip-H etc. significantly **outperform** SigLIP in local but significantly **underperform** SigLIP on test:
1. CLIP-ViT uses softmax as loss. It's learning objective is given **multiple** captions and classifiy them. This means while the model CAN, it does not learn fully the features useful for a single-caption task, which is what is test and not val.
2. SigLIP on the other hand, uses Sigmoid as loss, which operates on a one-to-one caption-image pair. This means the model is more suited for the task at hand, despite a smaller scale than H variants.
- Isolate the 2 tasks and evaluate separately on leaderboard: use pretrained SigLIP and iterate on YOLO until max, then turn to SigLIP.
- Large BS works a lot better for SigLIP as mentioned by many contrastive loss papers, due to the need for more negative samples in a batch.
- Iterating: train up to 5 epoch to validate helpfulness of change in hyperparams/augs then full train overnight
- 5 epoch proved enough for upscaled training data, 10epoch overfits.


### Evaluation (for finals)
Main change is switch to YOLOv6l6 as it shows better on own test set. Improvements were gained from training on full DSTA + own test set (denoted by "blind") below.
However, SAHI did not seem to help for yolov6l6, likely due to it being nearly its limit already.
Yolov6 is also faster than v9e

#### YOLOv6l6 augsv3 epoch29 blind conf=0.25 iou=0.3 + siglip-large-patch16-384 epoch5_cont_5_upscaled_augsv2
Same siglip as prev best of 0.905 on yolov9e
https://console.cloud.google.com/ai/platform/locations/asia-southeast1/batch-predictions/4075404169781968896?project=dsta-angelhack

test set:
- Accuracy: 0.908
- Speed Score: 0.7603945631481481


#### YOLOv6l6 augsv3 epoch22 + epoch29 blind conf=0.25 iou=0.3 WBF + siglip-large-patch16-384 epoch5_cont_5_upscaled_augsv2
epoch22 is not blind

test set:
- Accuracy: 0.909
- Speed Score: 0.7636044942592592

#### YOLOv6l6 augsv3 epoch 35 blind conf=0.25 iou=0.3 WBF + siglip-large-patch16-384 epoch5_cont_5_upscaled_augsv2
epoch35 is a new train run with dsta+own test as the previous run resume from checkpoint was broken

test set:
- Accuracy: 0.905
- Speed Score: 0.7485737964814815

#### YOLOv6l6 augsv3 epoch29 blind + epoch 35 blind conf=0.25 iou=0.3 WBF + siglip-large-patch16-384 epoch5_cont_5_upscaled_augsv2

test set:
- Accuracy: 0.91
- Speed Score: 0.7403886118518519
WBF of 0.905+0.908 = 0.91


#### YOLOv6l6 augsv3 epoch29 blind + epoch 35 blind conf=0.25 iou=0.3 WBF + siglip-large-patch16-384 epoch10v2
New trained from start with 10 epochs, same as epoch5v2 below (data is fully upscaled with pad=1 too, instead of nothing)

test set:
- Accuracy: 0.903
- Speed Score: 0.7448786192592592

Worse than epoch10v1


#### YOLOv6l6 augsv3 epoch29 blind + epoch 35 blind conf=0.5 iou=0.5 WBF 0.3 + siglip-large-patch16-384 epoch5v3
Note: conf=0.5 and 0.25 did not make any score difference in 0.91 submission, so it is safe to change that
v3 models are trained on expanded XS set (DSTA + own test, total 37120 boxes, where XS is 9214)

(wrong settings conf=0.5 iou=0.3 wbf=0.5):
- Accuracy: 0.913
- Speed Score: 0.7173836805555556

test set:
- Accuracy: 0.913
- Speed Score: 0.7431386035185186

#### YOLOv6l6 augsv3 epoch29 blind + epoch 35 blind conf=0.5 iou=0.5 WBF 0.3 + siglip-large-patch16-384 epoch10v3

(wrong settings conf=0.5 iou=0.3 wbf=0.5):
- Accuracy: 0.897
- Speed Score: 0.7440496585185186

test set:
- Accuracy: 0.897
- Score: 0.757444107037037

Siglip epoch 10v3 is overfitted

#### YOLOv6l6 augsv3 epoch29 blind + epoch 35 blind conf=0.5 iou=0.5 WBF 0.3 + siglip-so400m epoch5_merged
SO400m but trained on merged XS data

test set:
- Accuracy: 0.902
- Speed Score: 0.6077701553703703

Still worse than large

### Evaluation (only covers during qualifiers)

#### YOLOv9c 0.99 0.769 on own test
map@0.5 self calculated conf=0.1: 0.5095833333333334

pycocotools on conf=0.1:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.394
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.042
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.153
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.080
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.081
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.208
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.208
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.247
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.077
```

#### YOLOv9e 0.995 0.801 on own test
map@0.5 self calculated conf=0.1: 0.7375

pycocotools on conf=0.1:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.239
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.667
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.096
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.223
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.310
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.108
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.342
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.342
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.348
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.323
```

#### YOLOv9e 0.995 0.823 epoch65 on own test
map@0.5 self calculated conf=0.1: 0.7529166666666667

map@0.5 self calculated conf=0.5: 0.6794763513513513

pycocotools on conf=0.1:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.249
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.679
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.107
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.224
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.335
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.113
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.354
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.354
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.350
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.367
```

#### YOLOv9e 0.995 0.814 epoch89 augsv2 on own test
pycocotools on conf=0.1:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.197
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.574
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.073
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.184
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.256
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.100
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.289
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.289
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.296
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.264
```
Bad run.

#### YOLOv6l6 augsv3 on own test

| conf | IoU thresh | pycocotools mAP@0.5 | pycocotools mAP@0.5-0.95 | self calc mAP@0.5 over no. GT | self calc mAP@0.5 over no. preds |
|------|------------|---------------------|--------------------------|-------------------------------|----------------------------------|
| 0.1  | 0.5        | 0.806               | 0.319                    | 0.8606010016694491            | 0.47304427621013995              |
| 0.25 | 0.1        | 0.765               | 0.305                    | 0.800251256281407             | 0.8145780051150895               |
| 0.25 | 0.3        | 0.765               | 0.305                    | 0.8010887772194305            | 0.8095641134151502               |
| 0.25 | 0.5        | 0.773               | 0.307                    | 0.804857621440536             | 0.7998335414065751               |
| 0.25 | 0.7        | 0.772               | 0.309                    | 0.8115577889447236            | 0.7540856031128405               |
| 0.5  | 0.5        | 0.713               | 0.283                    | 0.7377946127946128            | 0.909704203425013                |


**Conclusion**: 
Yolov6l6's TPs are of lower conf than v9e which means a lower conf thresh must be used, this increases FPs. Lower IOU is more suitable for small obj det.
Overall, yolov6l6 outperforms all yolov9e. Self calculated mAP@0.5 over GT: v9e 0.6794763513513513 (conf=0.5, iou=0.1) vs v6l6 0.7377946127946128 (conf=0.5, iou=0.5). On cocotools map@0.5, v9e 0.679 vs v6l6 0.806 (conf=0.1, iou=0.5)
Choosing conf=0.25 and iou=0.3 for finals as it strikes balance between Precision and Recall on self calc.

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + siglip-large-patch16-256
test set:
- Accuracy: 0.675
- Speed Score: 0.7928535311111111


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + siglip-large-patch16-384 （**BEST on v9c**）
val set 0.8140147523709168

with upscale x4v3 pad=10: val set 0.40325029020448255

test set:
- Accuracy: 0.69
- Speed Score: 0.7744111877777777


#### YOLOv9e 0.995 0.801 iou=0.1 + siglip-large-patch16-384
0.414 is the F1 peak conf threshold

val set:
- conf=0.414 0.8535300316122234
- conf=0.365 0.8538812785388128
- conf=0.1 0.4345923743191356

test set:

conf=0.01:
- Accuracy: 0.778
- Speed Score: 0.7240670631481481
- Total 0.77260670631481481

conf=0.1:
- Accuracy: 0.777
- Speed Score: 0.7499369624074075
- Total 0.77429369624074075

conf=0.3:
- Accuracy: 0.776
- Speed Score: 0.7445673185185185
- Total 0.77285673185185185

**Conclusion** conf=0.1 best overall but in finals should use 0.01 as speed may not be as important


#### YOLOv9e 0.995 0.801 run last.pt iou=0.1 + siglip-large-patch16-384
val set:
- conf=0.3: 0.8559887600983491

test set:

conf=0.1
- Accuracy: 0.777
- Speed Score: 0.7499283801851853

with aug:
- Accuracy: 0.776
- Speed Score: 0.6983089774074074


#### YOLOv9e 0.995 0.825 epoch62 iou=0.1 + siglip-large-patch16-384
finetuned on 1600 input, 0.995map@0.5, 0.825map@0.5-0.95
conf=0.1: val set 0.8538812785388128

test set:

conf=0.1 aug:
- Accuracy: 0.78
- Speed Score: 0.7006760618518519

conf=0.1 no aug:
- Accuracy: 0.778
- Speed Score: 0.6996216127777778

TTA helps

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384
this epoch has better recall and lower bbox loss than epoch 62

conf=0.1: val set 0.8542325254654022

test set:

conf=0.01 aug:
- Accuracy: 0.782
- Speed Score: 0.680809122037037
- Total: 0.7718809122037037

conf=0.1 aug:
- Accuracy: 0.781
- Speed Score: 0.6885281759259259
- Total: 0.77175281759259259

conf=0.1 no aug:
- Accuracy: 0.779
- Speed Score: 0.753397417037037
- Total: 0.7764397417037037

conf=0.3 aug:
- Accuracy: 0.779
- Speed Score: 0.6993088687037037


both val and test slightly better than epoch62. TTA improves unlike other checkpoints. The improvement in score is not worth the time from augs for quals


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-epoch1
conf=0.1 val set 0.9694415173867229

test set:

conf=0.1 aug:
- Accuracy: 0.846
- Speed Score: 0.6200159431481481

**ALL SIGLIP FINETUNES ARE TRAINED ON FULL TRAIN SET, NO VAL SET**

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-epoch2
Train loss was high, likely due to initial lr to high (1e-4)
val set 0.7290129961362838

not testing


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-epoch3
5x3090, per gpu bs=12, no grad accum, effective bs 60

val set 0.9903407095187917

test set:

conf=0.1 aug:
- Accuracy: 0.823
- Speed Score: 0.6942834481481481

BS hurting it.


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-epoch4
5x3090, per gpu bs=12, no grad accum, effective bs 60

val set 0.9912188268352652

test set:

conf=0.1 aug:
- Accuracy: 0.79
- Speed Score: 0.7004231074074074

BS hurting it.

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-epoch10
5x3090, per gpu bs=12, no grad accum, effective bs 60

val set 0.9912188268352652

test set:

conf=0.1 aug:
- Accuracy: 0.836
- Speed Score: 0.7017661998148148

BS hurting it but training longer helps


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090
Models trained on 5x3090, per gpu bs=12, grad accum=16 = effective bs 960.

val set 0.9899894625922023

test set:

conf=0.1 aug:
- Accuracy: 0.864
- Speed Score: 0.6763070722222222

continued for 5 more epoch: (0.781 -> 0.874, 10 epoch ft = 0.093 map improvement)
- Accuracy: 0.874
- Speed Score: 0.6867920601851851

same (10epoch) but with real-esrgan x4v3 upscaling:
- Accuracy: 0.881
- Speed Score: 0.6629727242592593

Conclusion: Continued training does not reflect lower training loss but improves on test. Upscaling, despite being CONSISTENTLY bad locally, improves on test.

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-aug-epoch10
Models trained on 5x3090, per gpu bs=12, grad accum=16 = effective bs 960. With strong augs, final train loss 2.8 (high!)

```python
self.albu_transforms = A.Compose([
    A.GaussNoise(var_limit=2500, p=0.5),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Blur(p=0.1),
    A.ToGray(p=0.1),
    A.CLAHE(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.5, p=0.5),
    A.RandomGamma(p=0.2),
    A.Affine(scale=(0.8, 1.2), p=0.2),
    A.Perspective(p=0.5),
    A.ImageCompression(quality_lower=75, p=0.5),
    ToTensorV2()  # change back to CHW here
])
```

val set 0.662978573937478

test set:

conf=0.1 aug:
- Accuracy: 0.643
- Speed Score: 0.6780248025925926

Due to non normalized GaussianNoise, the model is screwed


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-aug-epoch10 fixed
Models trained on 5x3090, per gpu bs=12, grad accum=16 = effective bs 960. With FIXED augs (gaussian noise was not normalized)

```python
self.albu_transforms = A.Compose([
    A.GaussNoise(var_limit=2500/255/255, p=0.5),  # normalize
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Blur(p=0.1),
    A.ToGray(p=0.1),
    A.CLAHE(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.5, p=0.5),
    A.RandomGamma(p=0.2),
    A.Affine(scale=(0.8, 1.2), p=0.2),
    A.Perspective(p=0.5),
    A.ImageCompression(quality_lower=75, p=0.5),
    ToTensorV2()  # change back to CHW here
])
```

val set 0.7298911134527573

test set:

conf=0.1 aug:
- Accuracy: 0.674
- Speed Score: 0.6066626014814815


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-aug-`epoch3`0-v2
Models trained on 2x3090, per gpu bs=10, grad accum=48 = effective bs 960. With less augs. Final loss 2.4: https://wandb.ai/aliencaocao/TIL2024/runs/rc47xjod?nw=nwuseraliencaocao

```python
self.albu_transforms = A.Compose([
    A.GaussNoise(var_limit=500/255/255, p=0.5),  # normalize
    A.MultiplicativeNoise(p=0.5),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.CLAHE(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),
    A.RandomGamma(p=0.2),
    A.Perspective(p=0.5),
    A.ImageCompression(quality_lower=75, p=0.5),
    ToTensorV2()  # change back to CHW here
])
```

test set:

conf=0.1 aug:
- Accuracy: 0.672
- Speed Score: 0.6934455792592593

Conclusion: more epochs does not help, neither did less aug. Something else is fundamentally affecting the model

**FURTHER INVESTIGATION**:

torchvision ops operates on RGB while albumentations likely assumed BGR since it runs on top of OpenCV which uses BGR. This causes a swapped color channel but nothing will error out and model will still maintain some performance due to its robustness to color channel swaps. Reimplementing all torchvisison ops in albumentations made the loss normal.

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-aug-epoch10-fixed
This model is the same as the previous but with fixed augmentation due to torchvision/albumentations color channel mismatch. no weight decay. Resumed from epoch 5 (0.864)
```python
self.albu_transforms = A.Compose([
    A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
    A.GaussNoise(var_limit=400, p=0.5),
    A.MultiplicativeNoise(p=0.5),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.CLAHE(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),
    A.RandomGamma(p=0.2),
    A.Perspective(p=0.5),
    A.ImageCompression(quality_lower=75, p=0.5),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()  # CHW
])
```

test set:

conf=0.1 aug:
- Accuracy: 0.877
- Speed Score: 0.6770407942592593

Slight drop in perf VS 0.881 might be due to weight decay.



#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-so400m-patch14-384-ft-3090-epoch15-aug
with 1e-4 weight decay, on fixed aug:

```python
self.albu_transforms = A.Compose([
    A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
    A.GaussNoise(var_limit=500, p=0.5),
    A.ISONoise(p=0.5),
    A.MultiplicativeNoise(p=0.5),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.CLAHE(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),
    A.RandomGamma(p=0.2),
    A.Perspective(p=0.5),
    A.ImageCompression(quality_lower=75, p=0.5),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()  # CHW
])
```
val set 0.9706708816297858

test set:

conf=0.01 aug with upscale:
- Accuracy: 0.864
- Speed Score: 0.6026722651851852

conf=0.1 aug without upscale:
- Accuracy: 0.882
- Speed Score: 0.6314698722222223

conf=0.1 aug with upscale:
- Accuracy: 0.891
- Speed Score: 0.6100433083333334

own test V1: IoU@0.5: 0.5916666666666667

conf=0.3 aug with upscale:
- Accuracy: 0.891
- Speed Score: 0.623484457037037

Conclusion for conf-0.1 vs conf-0.3: 

previously with siglip-large, conf0.3 gave 0.779 while conf=0.1 gave 0.781. Here, we see equal performance.
This means that siglip-large has higher precision when given more bboxes containing more irrelevant images (and also more relevant images from the lowered conf).
so400m either wrongly classify these new relevant images as FN, which is supported by it having way more FPs at conf=0.01, or had just nice the same amount of new TPs and FNs that they cancel out. This can be a sign of it reaching its limit, meaning further improvement may have to come from obj detector.
In a way, so400m is worse than siglip large in preventing FPs, but it is also shown that without the obj detector spam BBOX, it is better at TPs and TNs. This places more emphasis that obj detector.

continued for 3 more epoch:

conf=0.1 aug with upscale:
- Accuracy: 0.877
- Speed Score: 0.6309745066666667

starting to overfit.

continue from epoch 15 with stronger augs for 4 more:
```python
A.GaussNoise(var_limit=1000, p=1.0),
A.ISONoise(p=1.0),
A.MultiplicativeNoise(p=1.0),
```
own test V2: 0.63875

test set:
- Accuracy: 0.889
- Speed Score: 0.6317016698148148

might be overfitting a bit


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-augv2_epoch5
Even stronger augs, rest same as best config (0.864 etc)
```python
self.albu_transforms = A.Compose([
    A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
    A.GaussNoise(var_limit=(500, 5000), p=1.0, per_channel=True),
    A.ISONoise(p=1.0, color_shift=(0.02, 0.07)),
    A.MultiplicativeNoise(p=1.0),
    A.AdvancedBlur(blur_limit=(3, 11), p=0.3),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.CLAHE(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),
    A.RandomGamma(p=0.2),
    A.Perspective(p=0.5),
    A.ImageCompression(quality_range=(25, 75), p=0.8),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()  # CHW
])
```

test set:

conf=0.1 aug with upscale:
- Accuracy: 0.884
- Speed Score: 0.6111272474074074

Compared to 0.881 for epoch 10 + fixed aug, this is just epoch 5 and proved to be better. New aug is good.


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-augv2_epoch5-upscaled
Same as above 0.884 but with training data upscaled 4x using real-esrgan x4v3 (pad=10 with image < 10 untouched)

own test V2: 0.6516666666666666

test set:

conf=0.1 aug with upscale pad=10:
- Accuracy: 0.892
- Speed Score: 0.5845252370370371

conf=0.1 aug with upscale pad=1:
- Accuracy: 0.887
- Speed Score: 0.6722072796296297

conf=0.1 aug with upscale pad conditional (>10: 10, <=10: 1):
- Accuracy: 0.894
- Speed Score: 0.6546273527777777

conf=0.5 aug with upscale pad conditional (>10: 10, <=10: 1):
- Accuracy: 0.899
- Speed Score: 0.6836933364814815

Training on upscale has significant benefit. It allows siglip large to outperform the current best 0.891 on SO400M 15 epoch aug with just 5 epoch. Equal perf on own test VS SO400M epoch19, a sign that own test V2 is correlating well to leaderboard.
pad=1 is bad for all BUT still better than not upscaling small images.
conf=0.5 is better now as new trained siglip-large is worse on low conf/many FP boxes

SAHI on epoch 65 with auto slice (6) conf=0.1:
- Accuracy: 0.867
- Speed Score: 0.2951938051851851

SAHI on epoch 65 with auto slice (6) conf=0.3:
- Accuracy: 0.891
- Speed Score: 0.3115894898148148

SAHI on epoch 65 with auto slice (6) conf=0.5:
- Accuracy: 0.902
- Speed Score: 0.31025421537037035

SAHI on epoch 65 with auto slice (6) conf=0.7:
- Accuracy: 0.884
- Speed Score: 0.3067358027777778

**Conclusion** SAHI need to use high conf thresh likely to reduce increased FPs as we zoom and slice

WBF with epoch62 and 65 at 1536  1, 1:
- Accuracy: 0.881
- Speed Score: 0.5903281548148147

WBF with epoch62 and 65 at 1600 1, 1:
- Accuracy: 0.885
- Speed Score: 0.5567839605555556

WBF with epoch62 and 65 at 1600 0.2, 1:
- Accuracy: 0.887
- Speed Score: 0.5553002087037037

The epoch 62 model dragging it down. Dropping reso to 1536 makes it worse.

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-augv2_epoch5-upscaled-v2
Same as above 0.884 but with training data upscaled 4x using real-esrgan x4v3 (pad=10 with image < 10 upscaled with pad=1)

test set:

no sahi conf=0.5:
- Accuracy: 0.89
- Speed Score: 0.6588264529629629

Worse than epoch5v1 upscaled cont 5ep below. Might be overfitting due to reduction in variance in training data by upscaling all images.

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-augv2_epoch5-upscaled-cont-5ep
Continued from above 0.894 for 5 more epoch but with training data upscaled 4x using real-esrgan x4v3 (pad=10 with image < 10 upscaled with pad=1)

No SAHI conf=0.1:
- Accuracy: 0.889
- Speed Score: 0.6697810924074075

This is worse than ep5 where (0.894 -> 0.889)

No SAHI conf=0.5:
- Accuracy: 0.896
- Speed Score: 0.6889453096296296

This is also worse than ep5 where (0.899 -> 0.896)

SAHI conf=0.5: **(BEST)**
- Accuracy: 0.905
- Speed Score: 0.28737872240740736

BUT this is better than ep5 where (0.902 -> 0.905)

`asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:yolo-ep65-aug-siglip-large-augv2-upscale-ep10-sahi`

`a6a5186814858bb9749e45fc9bd00cc83776616563c1ee5dedf76f79358f4158`

#### YOLOv9e 0.995 0.814 epoch89 iou=0.1 + siglip-large-patch16-384-augv2_epoch5-upscaled
New yolo trained on augsV2

SAHI conf0.5:
- Accuracy: 0.879
- Speed Score: 0.2790422940740741

On epoch5_cont_5 no SAHI conf0.5:
- Accuracy: 0.89
- Speed Score: 0.6602123187037037

new yolo bad, augs V2 sucks.

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-augv2_epoch5-upscaled-text-lock
Same as above but with text backbone frozen

test set:

conf=0.1 aug with upscale:
- Accuracy: 0.888
- Speed Score: 0.5911145787037038

**Conclusion**: previously shown that image backbone freezes are bad (0.864 -> 0.759 on SO400M). Text backbone however are only slightly worse (0.892 -> 0.888), likely because test data is MUCH more different on visuals than text descriptions compared to pretrained data. This can be a good way to finetune when compute-restricted.


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-so400m-patch14-384-augv2_epoch15-upscaled
Same augv2 and trained on upscaled data

test set:

conf=0.1 aug with upscale:
- Accuracy: 0.885
- Speed Score: 0.62944337

conf=0.5 aug with upscale:
- Accuracy: 0.892
- Speed Score: 0.6295357251851852

Unexpected but might be overfitting. Higher confidence improves which means it is weaker at FPs.


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-so400m-patch14-384-augv2_epoch5-upscaled
training loss is slightly lower than ep15, but may be due to shorter warmup period (10% of total steps).

conf=0.1 aug with upscale:
- Accuracy: 0.885
- Speed Score: 0.6373048718518519

weird but could be overfitting too since larger training data now means easier for model to learn.

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-so400m-patch14-384-augv2_epoch3-upscaled
conf=0.1 aug with upscale:
- Accuracy: 0.878
- Speed Score: 0.619298697037037

Underfitting. Weird that so400m is not doing very well on test set. Might be overfitting after all.


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-epoch15-aug
with 1e-4 weight decay, on fixed aug same as above so400m

own test V2: 0.64125

test set:

conf=0.1 aug with upscale:
- Accuracy: 0.887
- Speed Score: 0.5728528418518519

Conclusion: So400m outperform siglip-large on leaderboard but underperform on own test.


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384-ft-3090-epoch5-nodecay
Same as current best epoch 5 (0.864 test) but no weight decay

test set:
- Accuracy: 0.861
- Speed Score: 0.6977295988888889

No weight decay on whole thing doesnt help, the paper say use on text backbone only


#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-so400m-patch14-384-ft-3090-epoch5-nodecay-lit
siglip-so400m-patch14-384 with no decay, frozen vision, rest same as best (0.864) https://wandb.ai/aliencaocao/TIL2024/runs/odljiaec

test set:

conf=0.1 aug:
- Accuracy: 0.759
- Speed Score: 0.6347100637037038

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-so400m-patch14-384-ft-3090-epoch5-nodecay
same as above but with all unfrozen

test set:

conf=0.1 aug:
- Accuracy: 0.864
- Speed Score: 0.6515823366666667

with upscale:
- Accuracy: 0.878
- Speed Score: 0.6230587416666666

Same score but slower, not worth

#### YOLOv9e 0.995 0.823 epoch65 iou=0.1 + siglip-large-patch16-384
test set:

conf=0.1 aug:
- Accuracy: 0.782
- Speed Score: 0.648020966851852

best zero shot, marginally better than siglip-large-patch16-384 at speed cost


#### YOLOv9e 0.995 0.823 epoch67 iou=0.1 + siglip-large-patch16-384
this is the last checkpoint with lowest loss but likely overfitted

conf=0.1: val set 0.8551106427818757

test set

conf=0.1 aug:
- Accuracy: 0.778
- Speed Score: 0.7026901435185186

conf=0.1 no aug:
- Accuracy: 0.776
- Speed Score: 0.7372752572222223

val map slightly better than epoch65 but test is worse than epoch 62


#### YOLOv9c 0.99 0.769 conf=0.1 iou=0.1 + siglip-large-patch16-384
Recall @ 0.1 conf is 0.99+

val set:
- conf=0.1: 0.7987355110642782
- conf=0.365: 0.7987355110642782

test:

conf=0.01:
- Accuracy: 0.706
- Speed Score: 0.7528886066666667

conf=0.1:
- Accuracy: 0.705
- Speed Score: 0.7609215872222221

conf=0.365:
- Accuracy: 0.667
- Speed Score: 0.7041770444444444

conf=0.5:
- Accuracy: 0.678
- Speed Score: 0.7637064385185185

** Low conf reduce recall and increase FP but since CLIP is strong enough to identify the FPs it IMPROVES**


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + HuggingFaceM4/siglip-so400m-14-384-flash-attn2-navit
Same as siglip-so400m-14-980-flash-attn2-navit

val set 0.5319634703196348

not testing as poor perf on val set


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + openai/clip-vit-large-patch14
val set 0.8203371970495258

test set:
- Accuracy: 0.617
- Speed Score: 0.7981289131481482


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + openai/clip-vit-large-patch14-336
val set 0.8159466104671584

test set:
- Accuracy: 0.605
- Speed Score: 0.7732906711111112

high res degrade perf on val


#### YOLOv9c 0.99 0.769 iou=0.1 + CLIP-ViT-H-14-laion2B-s32B-b79K
val set 0.8767123287671232

with upscale x2 pad=10: val set 0.7600260841212911

with upscale x4v3 pad=1: val set 0.41789445486204124

with upscale x4v3 pad=10: val set 0.819634703196347

test set:

conf=0.365:
- Accuracy: 0.658
- Speed Score: 0.7836627077777778

conf=0.01:
- Accuracy: 0.654
- Speed Score: 0.7672438733333333

Upscaling still bad, ViT-H not as robust to FPs by YOLO as SigLIP-L

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
- Accuracy: 0.631
- Speed Score: 0.7869898768518518


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-b32-400m
val set 0.7695820161573587

not testing as clearly bad


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-b32-fullcc2.5b
val set 0.847383210396909

test:
- Accuracy: 0.635
- Speed Score: 0.8058501905555555

Conclusion: b16 better than b32 marginally


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-l14-400m
val set 0.8205128205128205

not testing as clearly worse than fullcc2.5b

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-l14-fullcc2.5b
val set 0.8640674394099052

with upscale x4v3 pad=10: val set 0.815595363540569

test set:
- Accuracy: 0.658
- Speed Score: 0.7976818785185185

#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + metaclip-h14-fullcc2.5b
val set 0.8739023533544081

with upscale x4v3 pad=10: val set 0.8243765367053039

test set:
- Accuracy: 0.64
- Speed Score: 0.7755046064814815

From val set only, H series has the best perf compared to L, L is better than B. More data lead to better.


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + ESRGANx2 + siglip-so400m-patch14-384
pre_pad=1: val set 0.672641652741911

pre_pad=10: val set 0.7766069546891464

Conclusion: ESRGANx2 is bad at prepad=1


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + ESRGANx4 + siglip-so400m-patch14-384
pre_pad=1: val set 0.7809975412715139

pre_pad=10: val set 0.720480753613773

Conclusion: ESRGANx4 is a lot better than 2x but 20% slower at prepad=1, ESRGANx4 is bad at prepad=10 somehow, but overall 4x is better than 2x


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + ESRGANx4v3 + siglip-so400m-patch14-384
x4v3 is the lite model, 100% faster!

pre_pad=1: val set 0.782051282051282

pre_pad=10: val set 0.7804706708816298

Conclusion: pre_pad 1 or 10 dont make much diff, but speed increase VS acc improvement is good. Still worse than without upscaling though. realesr-general-x4v3 is better than normal x4 marginally.

### Credits
SAHI batched inference implementation is modified from https://github.com/andressrodrl/sahi_custom/tree/batch_inf