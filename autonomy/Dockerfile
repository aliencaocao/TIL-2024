FROM python:3.8.15-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

WORKDIR $HOME
# install required robomaster libraries
RUN apt-get update && apt-get install curl libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx -y
RUN pip install -U pip
COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY src .

CMD uvicorn autonomy:app --port 5003 --host 0.0.0.0