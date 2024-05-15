FROM 12000sgd-nlp-base
# copy model
COPY src/models/gorilla-openfunctions-v2-5.0bpw-h6-exl2 models/gorilla-openfunctions-v2-5.0bpw-h6-exl2

# copy the rest of the files into the container
COPY src/api_service.py .
COPY src/NLPManager.py .

# start model service
CMD uvicorn api_service:app --port 5002 --host 0.0.0.0
