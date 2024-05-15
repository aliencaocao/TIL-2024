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
    apt-get install -qqy python3 python3-distutils python3-dev python3-pip libgl1-mesa-dev >/dev/null 2>&1

RUN pip install -U pip setuptools wheel
WORKDIR /workspace

# copy the rest of the files into the container
COPY src .

# copy requirements files
COPY requirements*.txt ./

# install mmdet requirements first
# otherwise requirements.txt will install wrong PyTorch version
RUN pip install -r requirements-pytorch.txt && \
    pip install -r requirements-mmcv.txt

RUN cd mmdetection && \
    pip install -v -e . && \
    pip install -r requirements/multimodal.txt && \
    pip install emoji ddd-dataset && \
    pip install git+https://github.com/lvis-dataset/lvis-api.git

# install other requirements
RUN pip install -r requirements.txt

RUN pip cache purge
RUN apt-get clean autoclean
RUN apt-get autoremove --yes
RUN rm -rf /var/lib/{apt,dpkg,cache,log}/

# start model service
CMD uvicorn api_service:app --port 5004 --host 0.0.0.0