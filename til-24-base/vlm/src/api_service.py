import base64
from fastapi import FastAPI, Request

from VLMManager import VLMManager


app = FastAPI()

vlm_manager = VLMManager()


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
        image_bytes = base64.b64decode(instance["b64"])

        bbox = vlm_manager.identify(image_bytes, instance["caption"])
        predictions.append(bbox)

    return {"predictions": predictions}
