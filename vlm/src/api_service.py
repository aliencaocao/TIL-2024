import base64
import os
from fastapi import FastAPI, Request

from VLMManager import VLMManager


app = FastAPI()

vlm_manager = VLMManager(
    # using old config because it works with new model on inference
    # while new config does not (because test set has no annotations)
    config_path="til-custom-config.py",
    weights_path=f"weights/{os.environ['WEIGHTS_FILENAME']}",
)


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/identify")
async def identify(instance: Request):
    """
    Performs Object Detection and Identification given an image frame and a text query.
    """
    # get base64 encoded string of image, convert back into bytes
    input_json = await instance.json()

    predictions = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        img_bytes = base64.b64decode(instance["b64"])

        bbox = vlm_manager.identify(img_bytes, instance["caption"])
        predictions.append(bbox)

    return {"predictions": predictions}
