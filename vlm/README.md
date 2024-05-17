# VLM

# Build & push container
```
docker build -t 12000sgd-vlm .

docker tag 12000sgd-vlm asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-vlm:latest

docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-vlm:latest

gcloud ai models upload --region asia-southeast1 --display-name '12000sgd-vlm' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-vlm:latest --container-health-route /health --container-predict-route /identify --container-ports 5004 --version-aliases default
```

# Run container
```
docker run -p 5004:5004 --gpus all 12000sgd-vlm
```

# Results
| Model | Step | LR schedule | Augmentation | Split `tokens_positive` by words? | AP50 | Speed |
| - | - | - | - | - | - | - |
| MM Grounding DINO | 8000 | Linear warmup 1k steps from 1e-5 to 1e-4, back to 1e-5 at 16800 steps | Default | Yes | 0.507 | 0.553 |
| MM Grounding DINO | 7000 | Linear warmup 1k steps from 1e-5 to 1e-4, back to 1e-5 at 16800 steps | Default | No | 0.488 | 0.548 |
| MM Grounding DINO | 5000 | Linear warmup 1k steps from 1e-5 to 1e-4, back to 1e-5 at 16800 steps | Default | No | Evaluating as of 170524 1553hrs | Evaluating as of 170524 1553hrs |

