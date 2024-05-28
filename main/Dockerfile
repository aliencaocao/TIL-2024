FROM python:3.8.15-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

WORKDIR $HOME
RUN pip install -U pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src .

CMD python participant_server.py