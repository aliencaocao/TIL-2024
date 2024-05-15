# Argument 1: path to JSONL input file
# Argument 2: path to JSON output file
# Argument 3: folder to look for images

import sys
import json
from PIL import Image
from tqdm import tqdm

caption_separator = ", "

infile_path, outfile_path, img_dir = sys.argv[1:]

with open(infile_path) as infile, open(outfile_path, "w") as outfile:
    for line in tqdm(infile):
        img_info = json.loads(line)
        filename, annotations = img_info["image"], img_info["annotations"]
        
        try:
            img = Image.open(f"{img_dir}/{filename}")
        except FileNotFoundError:
            continue

        regions = []
        cleaned_captions = []
        token_offset = 0
        for ann in annotations:
            # convert xywh to xyxy
            ann["bbox"][2] += ann["bbox"][0]
            ann["bbox"][3] += ann["bbox"][1]

            # throw away all chars other than alphabets and spaces
            cleaned_caption = "".join(ch for ch in ann["caption"] if ch == ' ' or ch.isalpha())
            
            regions.append({
                "bbox": ann["bbox"],
                "phrase": cleaned_caption,
                "tokens_positive": [[
                    token_offset,
                    token_offset + len(cleaned_caption),
                ]],
            })
            cleaned_captions.append(cleaned_caption)
            token_offset += len(cleaned_caption) + len(caption_separator)
        
        outfile.write(json.dumps({
            "filename": filename,
            "height": img.height,
            "width": img.width,
            "grounding": {
                "caption": caption_separator.join(cleaned_captions),
                "regions": regions,
            },
        }) + "\n")
