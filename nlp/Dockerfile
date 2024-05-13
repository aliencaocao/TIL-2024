FROM ubuntu:22.04
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
    apt-get install -qqy wget >/dev/null 2>&1

RUN pip install -U pip setuptools wheel
WORKDIR /workspace

# install other requirements
COPY requirements.txt .
COPY exllamav2-0.0.21-cp310-cp310-linux_x86_64.whl /exllamav2-0.0.21-cp310-cp310-linux_x86_64.whl
RUN pip install -r requirements.txt
RUN pip install /exllamav2-0.0.21-cp310-cp310-linux_x86_64.whl
RUN rm /exllamav2-0.0.21-cp310-cp310-linux_x86_64.whl

# copy model
COPY src/models/gorilla-openfunctions-v2-5.0bpw-h6-exl2 models/gorilla-openfunctions-v2-5.0bpw-h6-exl2

# copy the rest of the files into the container
COPY src/api_service.py .
COPY src/NLPManager.py .

RUN pip cache purge
RUN apt-get clean autoclean
RUN apt-get autoremove --yes
RUN rm -rf /var/lib/{apt,dpkg,cache,log}/

# start model service
CMD uvicorn api_service:app --port 5002 --host 0.0.0.0
