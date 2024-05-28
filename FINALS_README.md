# BrainHack TIL-24 Semis + Finals README

## Table of Contents

1. [Overview](#overview)
   1. [Additional Directories](#additional-directories)
   2. [Simulator](#simulator)
2. [Instructions](#instructions)
   1. [Local Developement Environment Setup](#local-developement-environment-setup)
   2. [Artifacts Submission](#artifacts-submission)
   3. [Docker + Artifact Registry Recap](#docker-and-artifact-registry-recap)
   4. [Docker Compose](#docker-compose)
3. [Local Testing](#local-testing)
4. [In-person Practice Session](#in-person-practice-session)
5. [Semi-Finals and Finals Flow](#semi-finals-and-finals-flow)
6. [Addtional Information](#addtional-information)
   1. [GCloud CLI](#gcloud-cli)
7. [Known Issues](#known-issues)
   1. [Robomaster SDK](#robomaster-sdk)

## Overview

### Additional Directories

You should see that the `til-24-base` repository now contains 3 additional directories:

- `main`, with template code for your main orchestration participant server to be deployed in the finals.
- `simulator`, with a testing version of the competition server simulator. You do NOT need to touch this folder at all.
- `autonomy`, with template code for your robotics autonomy code.

Note also that there are several new files in the top-level directory:

- `Dockerfile` for the test competition server,
- `docker-compose.yml` for connecting to your own competition server service for testing,
- `docker-compose-finals.yml` for connecting to the actual competition server (and thus not using your own competition server service),
- `test_competition_server.py`, the code for your testing version of the competition server, and
- an updated version of the `.env.example` file, which now includes additional variables for handling various networking configurations.

Take note that the `.env` file will be loaded by `docker compose`, and used to populate the environment variables the containers have access to. As such, be sure to update your `.env` with whichever condition you're testing. On Linux machines, the `host.docker.interal` hostname does not exist; instead, `172.17.0.1` is the IP address of the host from inside the container.

### Simulator

The simulator is written in JavaScript and uses `3js` to render the 3D environment of the turret. The environment is rendered as a cubemap surrounding the camera, which is positioned at the center of the cube and is given the ability to rotate around.

## Instructions

### Local Developement Environment Setup

From here on, you will need to have a local development environment set up for your own testing. This local development environment should have:

- Python3 (optional but highly recommended, an environment management system like `conda`)
- [Docker](https://www.docker.com/products/docker-desktop/)
- A modern web browser (e.g. Chrome, Firefox)

You will have to build two additional Docker images:

- `autonomy`, containing your robotics autonomy code responsible for controlling your Robomaster turret, and
- `main`, containing your main server orchestration code for the semi-finals + finals.

You will have to push these images to Artifact Registry (note: NOT Model Registry!), and these will be taken as your submissions along with your final model versions. Thus, though as a semi-finalist/finalist you will continue to have access to your Online Development Environment on GCP, this means you will likely want to set up the `gcloud` CLI on your local development environment as well if you want to be able to push containers from the same platform that you test them from; see [the installation instructions below](#gcloud-cli).

**_IMPORTANT NOTE: You should also run all your models simultaneously on your T4 instance on GCP to ensure that your models will all fit into the VRAM you will have access to during the semi-finals + finals (16GB of VRAM). Note that for the testing and finals, the network will be a LAN setup disconnected from the internet. As such, you must ensure that your images are able to be run offline_**

### Artifacts Submission

Participants should submit all 5 images tagged in the format `{[team-name]-[service]:finals}` to the GCP Artifact registry by 3 June 2024 12:00HRS. The 5 images are: **_ASR, NLP, autonomy, VLM and main_**.

Example of image tagging:

1. team-name-asr:finals
2. team-name-nlp:finals
3. team-name-vlm:finals
4. team-name-autonomy:finals
5. team-name-main:finals

### Docker and Artifact Registry Recap

```bash
# build container
docker build -t TEAM-NAME-autonomy .

# run container
docker run -p 5003:5003 TEAM-NAME-autonomy

# run container with GPU
docker run -p 5001:5001 --gpus all -d TEAM-NAME-asr

# view all running containers
docker ps

# view ALL containers, whether running or not
docker ps -a

# stop running container
docker kill CONTAINER-ID

# tag built image to be pushed to artifact registry
docker tag TEAM-NAME-asr asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-TEAM-NAME/TEAM-NAME-asr:finals

# push built image to artifact registry
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-TEAM-NAME/TEAM-NAME-asr:finals
```

When your model is pushed successfully, you should be able to see it under your team's repository on [Artifact Registry](https://console.cloud.google.com/artifacts/docker/dsta-angelhack/asia-southeast1).

### Docker Compose

To run all the containers, we will be using [Docker Compose](https://docs.docker.com/compose/), a tool that lets you easily configure and run multi-container applications.

```bash
# start all the services
docker compose up

# force a build of all services and start them afterwards
docker compose up --build

# take down all the services
docker compose down

# start a particular docker compose file by name (it defaults to `docker-compose.yml` if not indicated)
docker compose -f docker-compose-finals.yml up
```

## Local Testing

Create an `.env` file based on the provided `.env.example` file, and update it accordingly:

- `COMPETITION_IP = "172.17.0.1"` on Linux, `"host.docker.internal"` otherwise
- `LOCAL_IP = "172.17.0.1"` on Linux, `"host.docker.internal"` otherwise
- `USE_ROBOT = "false"`

Then run `docker compose up`. This should start the competition server locally, as well as the rest of the services accordingly to connect to it.

## In-person Testing Session

Note that for the testing setup, it will be a LAN setup disconnected from the internet. As such, you must ensure that your images are able to be run offline (i.e. have your model weights packaged in your image through your Dockerfile).

1. Your Docker images should be pre-loaded onto your desktop server hardware from Artifact Registry; verify that this is correct
2. Verify `.env` settings to connect to our competition server (should be correct, since we'll be setting this up for each machine in advance)
3. With either our `docker-compose-finals.yml` or your own custom `docker-compose.yml`, run `docker compose up`
4. We'll run the example test cases through your server simulating the semi-finals, and you can see how it all performs.
5. You can make any changes you need to during this time, but be sure to push them to Artifact Registry by the end of your session!

## Semi-Finals and Finals Flow

Note that for the finals setup, it will be a LAN setup disconnected from the internet. As such, you must ensure that your images are able to be run offline (i.e. have your model weights packaged in your image through your Dockerfile).

1. Your Docker images should be pre-loaded onto your desktop server hardware from Artifact Registry; verify that this is correct
2. Verify `.env` settings to connect to our competition server (should be correct, since we'll be setting this up for each machine in advance)
3. With your `docker-compose.yml` (could be based on our `docker-compose-finals.yml`), run `docker compose up`
4. We run our test cases through your server in a live stage setting, and you see whether you advance/win

## Addtional Information

### GCloud CLI

To install it, see the [installation docs provided by GCP](https://cloud.google.com/sdk/docs/install).

Then, run `gcloud init`, which should open a browser window for you to grant permissions for your user account. If prompted for your project id, input `dsta-angelhack`. If prompted for your region, input `asia-southeast1`. If prompted for your zone, input the zone corresponding to the zone of your team's instance (should be of the form `asia-southeast1-x` where x is any of `a`, `b`, or `c`).

## Known Issues

### Robomaster SDK

#### Python Version

For some reason, the python `robomaster` SDK provided by DJI only has pre-built wheels for python version 3.8 exactly. As such, your `autonomy` container likely needs to use exactly that python version. The existing template code uses a base image with a functioning version of python, but if you change the image, you will have to take note of this possible issue.

#### arm64

For some reason, the python `robomaster` SDK provided by DJI only has pre-built wheels for the `x86` architecture, entirely ignoring building `arm64` wheels for Macs on Apple Silicon. There is a workaround available using Rosetta and `conda` ([see this GitHub issue](https://github.com/dji-sdk/RoboMaster-SDK/issues/98)):

First, install `conda` from https://conda.io/projects/conda/en/latest/user-guide/install/macos.html and then run the following commands:

```bash
CONDA_SUBDIR=osx-64 conda create -n robo python=3.8.10
conda activate robo
python -c "import platform;print(platform.machine())" # x86_64
conda config --env --set subdir osx-64 # set future packages installs to also use osx-64 over arm64
pip install robomaster
```

This will create a new Python 3.8 environment `robo` and configure it to use `x86` packages using Rosetta instead of native ones for `arm64`.
