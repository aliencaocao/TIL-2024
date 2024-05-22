FROM 12000sgd-vlm-base

ENV WEIGHTS_FILENAME iter_12000_augs.pth

RUN mkdir weights
COPY weights weights

# start model service
CMD uvicorn api_service:app --port 5004 --host 0.0.0.0
# CMD sleep 100000
