import base64
import os
from fastapi import FastAPI, Request

from VLMManager import VLMManager

app = FastAPI()

vlm_manager = VLMManager(yolo_path='yolov9c_0.99_0.769.pt', siglip_path='siglip-so400m-patch14-384')


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
    img_bytes = [base64.b64decode(instance["b64"]) for instance in input_json["instances"]]
    captions = [instance["caption"] for instance in input_json["instances"]]
    bbox = vlm_manager.identify(img_bytes, captions)
    predictions.extend(bbox)

    return {"predictions": predictions}
