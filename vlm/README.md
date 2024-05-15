# VLM

# Build & push container
```
docker build -t 12000sgd-vlm .

docker tag 12000sgd-vlm asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-vlm:latest

docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-vlm:latest

gcloud ai models upload --region asia-southeast1 --display-name '12000sgd-vlm' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-vlm:latest --container-health-route /health --container-predict-route /identify --container-ports 5004 --version-aliases default

```