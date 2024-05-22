# ------------------- LOAD DATA -------------------

from datasets import Dataset
import orjson
from PIL import Image
from tqdm import tqdm

def get_split(split):
  ds_list = []
  with open(f"data/{split}.jsonl") as ann_file:
    for i, line in tqdm(enumerate(ann_file)):
      # {"image": "image_0.jpg", "annotations": [{"caption": "grey missile", "bbox": [912, 164, 48, 152]}, {"caption": "red, white, and blue light aircraft", "bbox": [1032, 80, 24, 28]}, {"caption": "green and black missile", "bbox": [704, 508, 76, 64]}, {"caption": "white and red helicopter", "bbox": [524, 116, 112, 48]}]}
      x = orjson.loads(line)
      anns = x["annotations"]
      img = Image.open(f"data/{split}/{x['image']}")
      # img.load() # bypass PIL lazy loading
      ds_list.append({
        "image_id": int(x["image"][6:-4]),
        # "image_id": i,
        "image": img,
        "width": img.width,
        "height": img.height,
        "objects": {
          # "id" key not used
          # "area": [ann["bbox"][2] * ann["bbox"][3] for ann in anns],
          "bbox": [ann["bbox"] for ann in anns],
          # TODO: categories aren't fixed. How to supply text?
          "caption": [ann["caption"] for ann in anns],
        },
      })

  return ds_list

train_ds = Dataset.from_list(get_split("train"))
val_ds = Dataset.from_list(get_split("val"))

# ------------------- TRANSFORMS -------------------

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

checkpoint = "google/owlv2-large-patch14-ensemble"

model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

import albumentations

augmentation = albumentations.Compose(
  transforms=[
    # How noisy will their test images be? var = 2500 is extreme!
    albumentations.GaussNoise(var_limit=2500, p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomBrightnessContrast(
      p=0.5,
      brightness_limit=0.3,
      contrast_limit=0.3,
    ),
    # MixUp / Mosaic?
  ],
  bbox_params=albumentations.BboxParams(format="coco", label_fields=["captions"]),
)

import numpy as np
import torch

def transform_sample(batch):
  # batch is a Dict[str, list], NOT a list!
  # perform augmentations HERE
  # but perform processing in collate_fn

  images_new = []
  objects_new = []

  for img, obj in zip(batch["image"], batch["objects"]):
    img_augmented = augmentation(
      image=np.asarray(img.convert("RGB")),
      bboxes=obj["bbox"],
      captions=obj["caption"],
    )
    images_new.append(img_augmented["image"])
    objects_new.append([
      {"bbox": bbox, "caption": caption}
      for bbox, caption in zip(img_augmented["bboxes"], img_augmented["captions"])
    ])

  batch["image"] = images_new
  batch["objects"] = objects_new
  return batch

def collate_fn(batch_list):
  batch = processor(
    text=[[obj["caption"] for obj in x["objects"]] for x in batch_list],
    images=[x["image"] for x in batch_list],
    return_tensors="pt",
  )

  # no need to put captions in labels because batch["input_ids"] contains them in tokenized form
  batch["labels"] = torch.nn.utils.rnn.pad_sequence(
    sequences=[torch.tensor([obj["bbox"] for obj in x["objects"]]) for x in batch_list],
    batch_first=True,
    padding_value=-1,
  )

  return batch

train_ds = train_ds.with_transform(transform_sample)
train_ds = val_ds.with_transform(transform_sample)

# ------------------- TRAIN -------------------

from transformers import TrainingArguments, Trainer

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")

    breakpoint()

    outputs = model(**inputs, return_dict=True)


    return (0, outputs) if return_outputs else 0
    

training_args = TrainingArguments(
  output_dir="owlv2-large-patch14-ensemble",
  num_train_epochs=30,
  learning_rate=5e-6,
  # lr_scheduler_type=schedule,
  eval_strategy="epoch",
  # auto_find_batch_size=True,
  # TODO: adjust batch size
  per_device_train_batch_size=1,
  per_device_eval_batch_size=1,
  save_strategy="epoch",
  bf16=True,
  dataloader_num_workers=1,
  remove_unused_columns=False,
  gradient_checkpointing=True,
)

trainer = CustomTrainer(
  model=model,
  args=training_args,
  train_dataset=train_ds,
  eval_dataset=val_ds,
  data_collator=collate_fn,
  tokenizer=processor,
)

trainer.train()