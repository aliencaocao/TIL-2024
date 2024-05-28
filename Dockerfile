FROM python:3.8.15-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

WORKDIR $HOME
RUN apt-get update && apt-get install curl -y
RUN pip install -U pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY simulator/ simulator/
COPY scoring/ scoring/
COPY test_competition_server.py .

CMD uvicorn test_competition_server:app --port 8000 --host 0.0.0.0