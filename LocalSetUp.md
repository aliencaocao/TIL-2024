# Installations


You will have to push these images to Artifact Registry (note: NOT Model Registry!), and these will be taken as your submissions along with your final model versions. Thus, though as a semi-finalist/finalist you will continue to have access to your Online Development Environment on GCP, this means you will likely want to set up the `gcloud` CLI on your local development environment as well if you want to be able to push containers from the same platform that you test them from; see [the installation instructions below](#gcloud-cli).

**_IMPORTANT NOTE: You should also run all your models simultaneously on your T4 instance on GCP to ensure that your models will all fit into the VRAM you will have access to during the semi-finals + finals (16GB of VRAM). Note that for the testing and finals, the network will be a LAN setup disconnected from the internet. As such, you must ensure that your images are able to be run offline_**


### GCloud CLI

To install it, see the [installation docs provided by GCP](https://cloud.google.com/sdk/docs/install).

Then, run `gcloud init`, 

Login to ur account (gmail)

Set ur project id to `dsta-angelhack`

`gcloud config set project dsta-angelhack`

Which is handled by the below

### Set up gcloud docker auth

`gcloud auth configure-docker asia-southeast1-docker.pkg.dev -q`

`gcloud config set artifacts/location asia-southeast1`

`gcloud config set artifacts/repository repository-$1`

### Install Required libraries
pip install -r requirements.txt

### Local Testing (Finals Environment)

Create an `.env` file based on the provided `.env.example` file, and update it accordingly:

- `COMPETITION_IP = "172.17.0.1"` on Linux, `"host.docker.internal"` otherwise
- `LOCAL_IP = "172.17.0.1"` on Linux, `"host.docker.internal"` otherwise
- `USE_ROBOT = "false"`

Then run `docker compose up`. This should start the competition server locally, as well as the rest of the services accordingly to connect to it.
Run this only when all the containers are built

# Models

Open `Docker Desktop`

## NLP
### Pulling
`docker pull asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:finals `

### Running
`docker run -p 5002:5002 --gpus all -d asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:finals`

## ASR

`TBC`

## VLM

`docker pull asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:yolo-ep65-aug-siglip-large-augv2-upscale-pad1-cond-ep5-conf0.1`

### Running 

`docker run -p 5003:5003 --gpus all -d asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-multistage-vlm:yolo-ep65-aug-siglip-large-augv2-upscale-pad1-cond-ep5-conf0.1`

## Kill all dockers

`docker kill $(docker ps -q)`


## Remove all images

`docker rm -v $(docker ps --filter status=exited -q)`


# Running 


`docker exec -it eeacabda5213 bash`

Sample copy

`docker cp eeacabda5213:4.jpg ./4.jpg`