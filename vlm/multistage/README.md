# Muli Stage approach
## Overview
1. YOLOv9 trained on single-class detection of targets in general
2. Extract the bboxes as detected by YOLO
3. Feed each bbox into a CLIP and get similarity score VS caption (1/image)
4. Choose the box with the highest similarity score

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

### Training YOLO
1. Initialize from YOLOv9e checkpoint
2. Train for 55 epochs with AdamW, lr=1e-3, effective bs=64, image size=1280, cosine LR schedule
3. Continue for 7 epochs with image size=1600 to improve on high-res and small object further since inference time we use 1600

### Training SigLIP using JAX
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
````shell
TFDS_DATA_DIR=/kaggle/input/til-siglip-tfds BV_JAX_INIT=1 python3 -m big_vision.trainers.proj.image_text.siglip --config big_vision/configs/proj/image_text/siglip_til.py --workdir til_train
```


### General takeaways
- Val and test correlation on CLIPs are not reliable beyond 0.8 mAP.
- Upscaling is always bad even though they do result in a clearer segregation of scores in CLIPs. To be investigated.
- SAHI (slicing inference) on YOLO is not suitable for this task despite it being designed for small objects detection.
- GaussianNoise augmentations significantly improve test scores


### Evaluation

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

conf=0.1 aug:
- Accuracy: 0.781
- Speed Score: 0.6885281759259259
- Total: 0.77175281759259259

conf=0.01 aug:
- Accuracy: 0.782
- Speed Score: 0.680809122037037
- Total: 0.7718809122037037

conf=0.3 aug:
- Accuracy: 0.779
- Speed Score: 0.6993088687037037

conf=0.1 no aug:
- Accuracy: 0.779
- Speed Score: 0.753397417037037
- Total: 0.7764397417037037

both val and test slightly better than epoch62. TTA improves unlike other checkpoints. The improvement in score is not worth the time from augs for quals


#### YOLOv9e 0.995 0.823 epoch67 iou=0.1 + siglip-large-patch16-384
this is the last checkpoint with lowest loss but likely overfitted

conf=0.1: val set 0.8551106427818757

test set

conf=0.1 aug:
- Accuracy: 0.778
- Speed Score: 0.7026901435185186

conf=0.1 no aug:
submitted but didnt receive score msg due to discord bot issue, but likely worse than epoch 62


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


#### YOLOv9c 0.99 0.769 conf=0.365 iou=0.1 + CLIP-ViT-H-14-laion2B-s32B-b79K
val set 0.8767123287671232

with upscale x2 pad=10: val set 0.7600260841212911

with upscale x4v3 pad=1: val set 0.41789445486204124

with upscale x4v3 pad=10: val set 0.819634703196347

test set:
- Accuracy: 0.658
- Speed Score: 0.7836627077777778

conf=0.01:
Accuracy: 0.654
Speed Score: 0.7672438733333333

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

**ESRGAN not worth it. Does not improve accuracy on all CLIPs.**


## TODO
1. Train YOLOv9c with 1600 resolution (now is 640 but infer at 1600 still helps)
2. Train YOLOv9c with noise augmentations
3. Manual impl slicing inference (batched) for YOLO to detect small objects, tried yolo-patched-inference and it sucks
4. Try speed diff of pipeline VS openclip
5. Eval YOLOv9c with augment=true on inference
6. Try google/siglip-large-patch16-256 on val
7. Try google/siglip-large-patch16-384 on val
