from typing import List
from mmdet.apis import DetInferencer
from PIL import Image
import numpy as np
import io

class VLMManager:
    def __init__(self, config_path, weights_path): 
        print("Config path:", config_path)
        print("Weights path:", weights_path)
        
        self.inferencer = DetInferencer(
            model=config_path,
            weights=weights_path,
            show_progress=False,
        )
        
        print("Model loaded.")

    def identify(self, img_bytes: bytes, caption: str) -> List[int]:
        # image is the raw bytes of a JPEG file
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.asarray(img)
                
        cleaned_caption = "".join(ch for ch in caption if ch == ' ' or ch.isalpha())
        
        # preds is a dict with keys 'labels', 'scores', 'bboxes'
        # bboxes are in xyxy format
        # results are sorted in descending order of confidence score
        preds = self.inferencer(
            img_array,
            texts=[cleaned_caption],
            tokens_positive=[[[0, len(cleaned_caption)]]],
        )["predictions"][0]
        
        if len(preds["labels"]) == 0:
            return [0, 0, 0, 0]
        
        # take bbox with highest score
        bbox = preds["bboxes"][0]
        
        # bbox is in xyxy format; convert to xywh
        bbox[2] -= bbox[0]
        bbox[3] -= bbox[1]
        return bbox
