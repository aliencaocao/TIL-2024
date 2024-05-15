from typing import List
from mmdet.apis import DetInferencer
from PIL import Image
import numpy as np
import io

class VLMManager:
    def __init__(self, config_path, weights_path):        
        self.inferencer = DetInferencer(
            model=config_path,
            weights=weights_path,
            show_progress=False,
        )

    def identify(self, img_bytes: bytes, caption: str) -> List[int]:
        # image is the raw bytes of a JPEG file
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.asarray(img)
                
        # throw away all chars other than alphabets and spaces
        cleaned_caption = "".join(ch for ch in caption if ch == ' ' or ch.isalpha())

        # word_start = 0
        # for i, ch in enumerate(cleaned_caption):
        #     if not ch.isalpha() or i == len(cleaned_caption)-1:
        #         tokens_positive.append([
        #             word_start,
        #             i + int(i == len(cleaned_caption)-1),
        #         ])
        #         word_start = i + 1
        
        results = self.inferencer(
            img_array,
            print_result=True,
            texts=cleaned_caption,
            tokens_positive=[0, len(cleaned_caption)],
        )
        
        breakpoint()

        return [0, 0, 0, 0]
