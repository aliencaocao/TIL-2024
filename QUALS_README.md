# BrainHack TIL-24 Base

## Important Links

- [Guardian's Handbook](https://www.notion.so/tribegroup/BrainHack-2024-TIL-AI-Guardian-s-Handbook-c5d4ec3c3bd04b0db0329884c220791f)
- [Educational Content on Google Drive](https://drive.google.com/drive/folders/1JmeEwQZoqobPmUeSZWrvR5Inrw6NJ8Kr)
- [Vertex AI Workbench on GCP](https://console.cloud.google.com/vertex-ai/workbench/instances?project=dsta-angelhack)

## First-time setup

Fork this repository on GitHub, and then clone your fork into the `$HOME` directory (should be `/home/jupyter`) of your GCP Vertex AI Workbench instance. Note that to clone your repository on the instance, you will likely need to [create a GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

Then from your instance, run the following command, replacing `TEAM-NAME` with your team name and `TRACK` with your track, either `novice` or `advanced`.

```bash
cd $HOME
bash init.bash TEAM-NAME TRACK
```

This should initialize [GCSFuse](https://cloud.google.com/storage/docs/gcs-fuse) to allow you to access your track's data GCS bucket (either `til-ai-24-novice` or `til-ai-24-advanced`, mounted at `$HOME/novice` or `$HOME/advanced` respectively), the NSC data bucket (`til-ai-24-data` mounted at `$HOME/nsc`) as well as your own team's private GCS bucket (`TEAM-NAME-til-ai-24` mounted at `$HOME/TEAM-NAME`) on your local filesystem. It should also configure Docker authentication to authenticate for the `asia-southeast1` Artifact Registry, as well as configure the default Docker repository to be your team's private repository (`repository-TEAM-NAME`).

WARNING: Do not run `init.bash` twice. While it shouldn't brick your instance, there's a chance it may. Run it once, with the right arguments, and everything will be fine.

## General notes

The Vertex AI Workbench is a full GPU-equipped JupyterLab environment, which should behave as you'd expect a typical Python + Jupyter environment to. You can start and stop your notebook instance from the [Google Cloud Platform UI](https://console.cloud.google.com/vertex-ai/workbench/instances?project=dsta-angelhack). While the instances will automatically shut down after 30 minutes of inactivity, it is preferred that you shut off your instances yourselves after you are done with them to reduce the costs associated with your compute. We also recommend that you take advantage of the built-in GitHub integration and regularly push your code to your own branch, such that if anything goes wrong with the development environment, your code is still readily accessible. Similarly, consider storing important checkpoints/model weights in Google Cloud Storage.

In this competition we will leverage Docker extensively to ensure that models are able to be consistently deployed for the finals.

### Repository structure

Broadly, this repository is split into several subdirectories based on the tasks in the competition.

- `asr`: For the automatic speech recognition (ASR) task.
- `nlp`: For the natural language processing (NLP) task.
- `vlm`: For the object detection by vision-language model (VLM) task.
- `scoring`: directory containing the evaluation functions we will be using to evaluate your models. You should NOT modify anything in here. If you notice an issue with any of the `scoring` functions, please bring it up with an AngelHack or DSTA staff member.

Note also several example test scripts made available for testing your ASR, NLP, and VLM models; see the `test_asr.py`, `test_nlp.py`, and `test_vlm.py` files, respectively.

### Compute Availability

The online development environment gives each team access to a Linux virtual machine with:

- [n1-standard-4](https://cloud.google.com/compute/docs/general-purpose-machines#n1_machine_types) (4 cores, 15 GB RAM)
- Nvidia Tesla T4 (16 GB VRAM)
- 200 GB Disk space

In the Semi-Finals and Finals, teams will have access to a Linux machine with the following:

- Intel Core i7-14700K
- 16 GB RAM
- Nvidia RTX 4070 Ti Super (16 GB VRAM)
- 1 TB Disk space
- Ubuntu 20.04

Note that this means you should not be finding the absolute largest possible models to train and submit in the Virtual Qualifiers, as you may be unable to run all your models simultaneously on the hardware available to you later on in the Semi-Finals/Finals. Be sure to test your submitted models to ensure they will all fit into the available VRAM of your machine, and plan around this limitation accordingly!

## Data Augmentation

In real-life scenarios, particularly in machine learning applications that operate in dynamic and diverse environments, it is common for the distribution of incoming deployment data to deviate from the distribution of the data the model was trained on. As a result of data drift, your training data will not perfectly match the data presented to you in subsequent rounds. In light of this, you will have to ensure your submitted models are robust enough to still make good predictions on noisy out-of-distribution data.

While your team is free to use whatever libraries you wish to use for data augmentation, we suggest the [`audiomentations`](https://iver56.github.io/audiomentations/) library for audio augmentation and the [`albumentations`](https://albumentations.ai/docs/) library for image augmentation due to their ease of use.

### Dataset Details

While Operation Guardian seeks to deploy a cutting-edge defence system across the nation through the use of autonomous air defence turrets to identify, classify and take down air threats, numerous obstacles stand in the path of the Guardians working to develop the AI algorithm.

In a fortified outpost nestled amidst rugged terrains, the seamless flow of communication between the Command and Control Centre (C2) and turret is starting to falter, plagued by the creep of corruption within the radio transmissions. Each command issued by the control room is met with a barrage of **static radio communications noise**, rendering the instructions barely inaudible. The relentless march of time only serves to exacerbate the issue, as **the noise within the radio transmissions worsens with each passing day**.

Furthermore, as the robots mounted atop the turrets have been in use for quite some time, the robot sensors are starting to exhibit signs of degradation, with a growing veil of **digital noise** cast over its view of the world. With each passing day, the cacophony of distortions is forecasted to intensify, obscuring the once-clear images perceived by the sensor. While half of the robots which have been designated to the Novice Guardians have already been replaced by brand new equipment, **those assigned to the Advanced Guardians are still awaiting replacement**. Consequently, the **Advanced Guardians have to adapt to increasingly noisy visual images** and navigate through the challenges posed by their deteriorating sensors.

#### Misc Technical Details

Audio files are provided in .WAV format with a sample rate of 16 kHz. Images are provided as 1520x870 JPG files.

In the audio datasets provided to both the Novice and Advanced Guardians, noise will already be present. Should Guardians wish to finetune their models on additional data, they are also free to use the (clean, unaugmented) National Speech Corpus data present in the til-ai-24-data bucket.

As for the image datasets provided to both the Novice and Advanced Guardians, there will be no noise present. However, it is worth noting that Advanced Guardians' models would have to be adequately robust to noise due to the degradation of their robot sensors.

### Audio Augmentation with audiomentations

First, install the `audiomentations` library using `pip`. Note that you wouldn't need it in your model submission container since this is just for your model training, so you don't need to add it to your `requirements.txt`.

```bash
pip install audiomentations
```

[The audiomentation docs](https://iver56.github.io/audiomentations/) give examples of how to use the `audiomentations` library. In general, so long as you've loaded your audio files as a `numpy` array, you should be able to pass it into the `augment` function created by `audiomentations.Compose`. Be sure to look through the documentation to see all the different augmentations available.

### Image Augmentation

First, install the `albumentations` library using `pip`. Note that you wouldn't need it in your model submission container since this is just for the training phase, so you don't need to add it to your `requirements.txt`.

```bash
pip install albumentations
```

Then you should follow along with the [albumentations guide for bounding box augmentation](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/). Note in particular that bounding box coordinates are provided in the COCO format i.e. `[x_min, y_min, width, height]`, sometimes elsewhere referred to as LTWH for left top width height.

## Submission Instructions

Teams are advised to submit early; don't put it off all the way until the end! Note that the leaderboard reflects your latest submission, not your best all-time submission. While you are encouraged to experiment and make several submissions, your progression onto the semi-finals and finals will be based on the leaderboard positions after your team's final submissions before the submission deadline. The responsibility is on you and your team to ensure that the results from your best models are what will be used to determine whether you progress.

Teams are also recommended to test their Docker containers prior to submission using the provided test scripts (i.e. `test_asr.py`, `test_nlp.py`, `test_vlm.py`) to ensure that most of the common bugs regarding hitting your container endpoints are resolved first. This also allows teams to improve their own speed of iteration as submission evaluation can take upwards of an hour.

### Building a container

To build your container, run the following command from where your Dockerfile is located, which should be the directory of your container (e.g. `/asr`). Be sure to also update the tag (the `-t` flag) with your team name and the model task (`asr`, `nlp`, `vlm`) accordingly.

```bash
docker build -t TEAM-NAME-asr .
```

Note that while you _can_ tag your containers anything you want, we recommend you name it something like `{TEAM-NAME}-{TASK}`, and all examples will follow this naming convention.

### Testing your container docker with GPU

As an example, here we run the `TEAM-NAME-asr` container built at the previous step. Be sure to update the ports based on which ports need to be exposed from your container (as per your Dockerfile). The `--gpus all` flag gives the container access to all GPUs available to the instance, and the `-d` flag runs the container in detached mode in the background.

```bash
docker run -p 5001:5001 --gpus all -d TEAM-NAME-asr
```

To view all running containers, run:

```bash
docker ps
```

To stop a container, run the following command (where `CONTAINER-ID` is the container ID you can find from `docker ps`, like `a1b2c3b4` ):

```bash
docker kill CONTAINER-ID
```

For other useful Docker commands, see this [Docker cheat sheet](https://docs.docker.com/get-started/docker_cheatsheet.pdf).

### Submitting your models

Model submission is being handled entirely through GCP as well. You first push your built Docker containers to Artifact Registry, then upload the model to the Vertex AI Model Registry.

#### Push to Artifact Registry

You tag your container (`TEAM-NAME-asr` in the below example) locally with your remote repository (`repository-TEAM-NAME` below) and the artifact+tag (`TEAM-NAME-asr:latest` in the example) you want to push to. Then you run `docker push` to actually push.

```bash
docker tag TEAM-NAME-asr asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-TEAM-NAME/TEAM-NAME-asr:latest
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-TEAM-NAME/TEAM-NAME-asr:latest
```

When your model is pushed successfully, you should be able to see it under your team's repository on [Artifact Registry](https://console.cloud.google.com/artifacts/docker/dsta-angelhack/asia-southeast1).

#### Upload to Model Registry for Submission

Once your model is on Artifact Registry, you can submit it by uploading it to Model Registry. Note that the subsequent automated model evaluation depends on the `--display-name` parameter, where the last 3 characters (`asr` in the following example) determine which task the model is meant to accomplish. The possible options as `asr`, `nlp`, and `vlm`. We suggest the same format as suggested before, `{TEAM-NAME}-{TASK}`.

Take note to update the flags `--container-health-route`, `--container-predict-route`, `--container-ports`, etc., which describe how our automation can interact with your container. For more, see the GCP Vertex AI documentation on [importing models programmatically](https://cloud.google.com/vertex-ai/docs/model-registry/import-model#import_a_model_programmatically) and all [the optional flags](https://cloud.google.com/sdk/gcloud/reference/ai/models/upload#OPTIONAL-FLAGS). The `health-route`should accept GET requests and return `200 OK` if the container is ready to receive prediction requests. The `predict-route` should accept POST requests and actually handle inference/prediction. The `ports` should be a comma-separated list (or a single value) of the ports your container has exposed.

```bash
gcloud ai models upload --region asia-southeast1 --display-name 'TEAM-NAME-asr' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-TEAM-NAME/TEAM-NAME-asr:latest --container-health-route /health --container-predict-route /stt --container-ports 5001 --version-aliases default
```

Shortly after successfully running the command, you should receive a Discord notification providing a link to review the status of a batch prediction job which evalulates your model accuracy and speed. If you do not, ping `@alittleclarity` on the BH24 TIL-AI Discord server in your team's private channel.

When the job finishes, which could take anywhere from 20 minutes to an hour, your model results will be updated on the leaderboard and you should be notified accordingly in your team's private Discord channel. If you don't get any updates, check the status on your batch prediction link. Note that during periods of high traffic/demand, it is likely that your job will take longer.

## Known Issues

### Development Environment

#### init.bash

Don't run `init.bash` more than once. If your initialization doesn't work properly, ping `@alittleclarity` on the BH24 TIL-AI Discord server in your team's private channel.

#### GCP Auth Credentials

By default, the Vertex AI instances are set up to use the auth credentials of your team's GCP service account, which all the subsequent automations and permissions depends on. You can check which credentials you are using by running `gcloud auth list`. If you've set up additional credentials (e.g. by using `gcloud auth login`), your model submission commands may not work properly. You can switch which credentials are active using the `gcloud config set account SERVICE-ACCOUNT-EMAIL@dsta-angelhack.iam.gserviceaccount.com` command, where `SERVICE-ACCOUNT-EMAIL` is the email of your service account (which should be either `svc-TEAM-NAME` or `sa-TEAM-NAME`).

#### Loading JupyterLab

Sometimes loading the JupyterLab environment can fail or get stuck. Forcing a hard browser refresh (`Ctrl-Shift-R` on Windows or `Cmd-Shift-R` on Mac) on the JupyterLab browser page usually solves this.

#### NVIDIA Driver Not Recognized

Sometimes the NVIDIA drivers stop being recognized on the instance. You can see this when you run the `nvidia-smi` command. If this happens, ping @alittleclarity on Discord, but here are some steps you can follow to possibly resolve this issue.

```bash
sudo apt-get purge nvidia-*
sudo apt-get update
sudo apt-get autoremove
```

Then stop and start the instance.

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-debian10-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo dpkg -i cuda-repo-debian10-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo cp /var/cuda-repo-debian10-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
sudo apt-get install -y cuda-drivers
```

See also:

- https://www.googlecloudcommunity.com/gc/Infrastructure-Compute-Storage/A100-GPU-VM-on-GCP-NVIDIA-SMI-has-failed-because-it-couldn-t/m-p/480629
- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=10&target_type=deb_local

### Libraries

Versions of the HuggingFace transformers library `>4.37.0` may not work due to issues with the `torch_xla` library on GCP instances. Unless you have a specific need otherwise, we recommend `transformers==4.37.0` if you intend on using the library.
