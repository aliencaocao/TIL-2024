import base64

from fastapi import FastAPI, Request

from VLMManager import VLMManager

app = FastAPI()

vlm_manager = VLMManager(
    yolo_paths=[
        # '29_ckpt_yolov6l6_blind.pt',
        # '35_ckpt_yolov6l6_blind_run2.pt',
        '68_ckpt_yolov6l6_blind_dstaxs7444_trt.pth',
    ],
    
    # Ignored if non-YOLOv6 or non-SAHI.
    # If SAHI but not YOLOv6, set same path as in yolo_paths.
    sliced_yolo_paths=[
        # '29_ckpt_yolov6l6_blind.pt',
        # '35_ckpt_yolov6l6_blind_run2.pt',
        '68_ckpt_yolov6l6_blind_dstaxs7444_trt.pth',
    ],

    clip_path='siglip_model_epoch5v3_merged',
    upscaler_path='realesr-general-x4v3.pth',
    use_sahi=[False],
    siglip_trt=True,  # set to True if using TensorRT SigLIP for finals
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
    img_bytes = [base64.b64decode(instance["b64"]) for instance in input_json["instances"]]
    captions = [instance["caption"] for instance in input_json["instances"]]
    bbox = vlm_manager.identify(img_bytes, captions)
    predictions.extend(bbox)

    return {"predictions": predictions}
