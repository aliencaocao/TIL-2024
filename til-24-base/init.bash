#!/bin/bash
if [ -z "$1" ];
    then echo "team name not set";
    echo "usage: bash init.bash TEAM-NAME TRACK"
    echo "example: bash init.bash test-team novice"
    exit 1;
fi
if [ -z "$2" ];
    then echo "competition track not set";
    echo "usage: bash init.bash TEAM-NAME TRACK"
    echo "example: bash init.bash test-team novice"
    exit 1;
fi

# create track dir, either novice or advanced
mkdir $HOME/$2

# create team dir
mkdir $HOME/$1

mkdir $HOME/nsc

# add bucket mount settings to /etc/fstab
echo til-ai-24-$2 $HOME/$2 gcsfuse ro,allow_other,implicit_dirs,uid=1000,gid=1001,_netdev | sudo tee -a /etc/fstab > /dev/null
echo $1-til-ai-24 $HOME/$1 gcsfuse rw,allow_other,implicit_dirs,uid=1000,gid=1001,_netdev | sudo tee -a /etc/fstab > /dev/null
echo til-ai-24-data $HOME/nsc gcsfuse ro,allow_other,implicit_dirs,uid=1000,gid=1001,_netdev | sudo tee -a /etc/fstab > /dev/null

# add user_allow_other to /etc/fuse.conf
echo user_allow_other | sudo tee -a /etc/fuse.conf > /dev/null

# mount with linux
sudo mount $HOME/$2
sudo mount $HOME/$1
sudo mount $HOME/nsc

# set up gcloud docker auth
gcloud auth configure-docker asia-southeast1-docker.pkg.dev -q
gcloud config set artifacts/location asia-southeast1
gcloud config set artifacts/repository repository-$1

# install required libraries
pip install -r requirements.txt

# create .env
echo "TEAM_NAME=$1\nTEAM_TRACK=$2" > .env