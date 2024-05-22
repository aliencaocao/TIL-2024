# VLM

# Build base
```bash
docker build -t 12000sgd-vlm-base -f base.dockerfile .
```

## Build & push container
```bash
docker build -t 12000sgd-vlm . && \

docker tag 12000sgd-vlm asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-vlm:latest && \

docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-vlm:latest && \

gcloud ai models upload --region asia-southeast1 --display-name '12000sgd-vlm' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-vlm:latest --container-health-route /health --container-predict-route /identify --container-ports 5004 --version-aliases default
```

## Run container
```bash
docker run -p 5004:5004 --gpus all 12000sgd-vlm
```

## Results
| Model | Step | LR schedule | Train aug | Test aug | Split `tokens_positive` by words? | AP50 | Speed |
| - | - | - | - | - | - | - | - |
| MM Grounding DINO | 8000 | Linear warmup 1k steps from 1e-5 to 1e-4, back to 1e-5 at 16800 steps | Default | Default | Yes | 0.507 | 0.5533085946296297 |
| MM Grounding DINO | 7000 | Linear warmup 1k steps from 1e-5 to 1e-4, back to 1e-5 at 5000 steps | Default | Default | No | 0.488 | 0.5482950594444445 |
| MM Grounding DINO | 5000 | Linear warmup 1k steps from 1e-5 to 1e-4, back to 1e-5 at 5000 steps | Default | Default | No | 0.479 | 0.5338830794444445 |
| MM Grounding DINO | 9000 | Linear warmup 1k steps from 1e-5 to 1e-4, back to 1e-5 at 5000 steps | Default | Default | No | 0.487 | 0.5549248251851852 |
| MM Grounding DINO | 10000 | Linear warmup 1k steps from 3.95e-5 to 3.95e-4, back to 3.95e-5 at 5000 steps | Brightness, contrast, Gaussian noise | Default | No | 0.509 | 0.4668306538888889 |
| MM Grounding DINO | 10000 | Same as prev, but trained for additional 2000 steps with LR = 3.95e-6 | Brightness, contrast, Gaussian noise | Default | No | 0.504 | 0.564885440925926 |
