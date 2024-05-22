FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu122.py310
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "verbose = off" > ~/.wgetrc

# override the US mirrors as they are too slow
COPY sources.list /etc/apt/sources.list

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# pip gives a warning if you install packages as root
# set this flag to just ignore the warning
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get -qq update >/dev/null 2>&1 && \
    apt-get -qq upgrade -y >/dev/null 2>&1 && \
    apt-get install -qqy python3 python3-distutils python3-dev python3-pip libgl1 >/dev/null 2>&1

WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm requirements.txt

COPY siglip-so400m-patch14-384 siglip-so400m-patch14-384
COPY yolov9c_0.99_0.769.pt yolov9c_0.99_0.769.pt

COPY api_service.py api_service.py
COPY VLMManager.py VLMManager.py

# clear cache and unnecessary files
RUN pip cache purge
RUN apt-get clean autoclean
RUN apt-get autoremove --yes
RUN rm -rf /var/lib/{apt,dpkg,cache,log}/

CMD uvicorn api_service:app --port 5004 --host 0.0.0.0
