# Argument 1: path to JSONL input file
# Argument 2: path to JSON output file

import sys
import json
from PIL import Image

vg_annotations = []

caption_separator = ", "

with open(sys.argv[1]) as jsonl_infile:
    for line in jsonl_infile:
        img_info = json.loads(line)
        filename, annotations = f"til/images/{img_info['image']}", img_info["annotations"]

        img = Image.open(filename)

        regions = []
        cleaned_captions = []
        token_offset = 0
        for ann in annotations:
            # convert xywh to xyxy
            ann["bbox"][2] += ann["bbox"][0]
            ann["bbox"][3] += ann["bbox"][1]

            tokens_positive = []
            # throw away all chars other than alphabets and spaces
            cleaned_caption = "".join(ch for ch in ann["caption"] if ch == ' ' or ch.isalpha())
            word_start = 0
            for i, ch in enumerate(cleaned_caption):
                if not ch.isalpha() or i == len(cleaned_caption)-1:
                    tokens_positive.append([
                        word_start + token_offset,
                        i + token_offset + int(i == len(cleaned_caption)-1)
                    ])
                    word_start = i + 1
            
            regions.append({
                "bbox": ann["bbox"],
                "phrase": cleaned_caption,
                # "tokens_positive" field may be unnecessary?
                # see https://github.com/longzw1997/Open-GroundingDino/blob/main/data_format.md
                "tokens_positive": tokens_positive,
            })
            cleaned_captions.append(cleaned_caption)
            token_offset += len(cleaned_caption) + len(caption_separator)
        
        vg_annotations.append({
            "filename": filename,
            "height": img.height,
            "width": img.width,
            "grounding": {
                "caption": caption_separator.join(cleaned_captions),
                "regions": regions,
            },
        })
        
with open(sys.argv[2], "w") as vg_outfile:
    json.dump(vg_annotations, vg_outfile)

# import json
                    
# with open("vlm.jsonl") as jsonl:
#     num_comma_anns = 0
#     for i, line in enumerate(jsonl):        
#         img_info = json.loads(line)
#         filename, annotations = img_info["image"], img_info["annotations"]
        
#         for j, ann in enumerate(annotations):
#             if "," in ann["caption"]:
#                 print(f"Comma found in line {i+1}, caption \'{ann['caption']}\'")
#                 num_comma_anns += 1
    
#     print(f"{num_comma_anns} captions contain commas.")
