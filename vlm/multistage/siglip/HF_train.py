import os

os.environ["WANDB_PROJECT"] = "TIL2024"

import torch
from datasets import load_dataset
import cv2
import sys
from torchvision.io import ImageReadMode, read_image
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoImageProcessor,
    AutoProcessor,
    EvalPrediction
)
import numpy as np
from modeling_siglip import SiglipModel
import albumentations as A
from albumentations.pytorch import ToTensorV2

pretrained_model_path = "google/siglip-large-patch16-384"

model = SiglipModel.from_pretrained(pretrained_model_path)
processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-384")  # always use pretrained
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
# image_processor = AutoImageProcessor.from_pretrained(pretrained_model_path)
config = model.config


# split the ds into train and val. the image column is in format "image_1234_1.jpg". Take 1234 as the image id. Check if the id is smaller than 4086. If yes, put this in train, else in val.
def get_ds():
    # If error about missing name, use datasets==2.15.0
    dataset = load_dataset(path='.', data_files="til_siglip_ds.json")
    # train_dataset = dataset.filter(lambda example: int(example["image"].split("_")[1]) < 4086)
    # val_dataset = dataset.filter(lambda example: int(example["image"].split("_")[1]) >= 4086)
    # return train_dataset, val_dataset
    return dataset


# train_dataset, val_dataset = get_ds()
train_dataset = get_ds()


class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.albu_transforms = A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
            A.GaussNoise(var_limit=(500, 5000), p=1.0, per_channel=True),
            A.ISONoise(p=1.0, color_shift=(0.02, 0.07)),
            A.MultiplicativeNoise(p=1.0),
            A.AdvancedBlur(blur_limit=(3, 11), p=0.3),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.CLAHE(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),
            A.RandomGamma(p=0.2),
            A.Perspective(p=0.5),
            A.ImageCompression(quality_range=(25, 75), p=0.8),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()  # CHW
        ])

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = np.asarray(x.permute(1, 2, 0), dtype=np.uint8)  # torch CHW to HWC as required by albumentations
            x = self.albu_transforms(image=x)['image']
            x = x.float()
        return x


# For preprocessing the datasets.
# Initialize torchvision transforms and jit it for faster processing.
image_transformations = Transform(
    config.vision_config.image_size, processor.image_processor.image_mean, processor.image_processor.image_std
)

# image_transformations = torch.jit.script(image_transformations)

def preprocess_dataset(dataset):
    # Preprocessing the datasets.
    data = dataset['train']
    # We need to tokenize inputs and targets.
    column_names = data.column_names

    # 6. Get the column names for input/target.
    image_column = "image"
    caption_column = "label"

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = processor.tokenizer(captions, padding="max_length")
        examples["input_ids"] = text_inputs.input_ids
        return examples

    data = data.map(
        function=tokenize_captions,
        batched=True,
        remove_columns=[col for col in column_names if col != image_column],
    )

    def transform_images(examples):
        image_dir = 'til_siglip_ds_x4v3_v2/'
        images = [read_image(image_dir + image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    # Transform images on the fly as doing it on the whole dataset takes too much time.
    data.set_transform(transform_images)
    return data


train_dataset = preprocess_dataset(train_dataset)
# val_dataset = preprocess_dataset(val_dataset)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "return_loss": True,
    }

from sklearn.metrics import f1_score
def compute_metrics(pred: EvalPrediction):
    predictions = pred.predictions
    labels = pred.label_ids
    return {"accuracy": (predictions == labels).mean().item(), 'F1': f1_score(labels, predictions, average='macro')}


effective_bs = 960
gpu_num = 5
per_dev_bs = 12  # can increase when got more GPUs due to FDSP
grad_accum = int(960 / gpu_num / per_dev_bs)

# initialize Trainer
training_args = TrainingArguments(
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=1e-4,
    per_device_train_batch_size=per_dev_bs,
    remove_unused_columns=False,
    output_dir="siglip-finetune",
    # per_device_eval_batch_size=1,
    gradient_accumulation_steps=grad_accum,
    adam_beta1=0.9,
    adam_beta2=0.99,  # decrease from 0.999
    num_train_epochs=10,
    lr_scheduler_type="cosine",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=0.9999,
    # eval_strategy="epoch",  # eval bugged causing oom
    save_total_limit=5,
    bf16=torch.cuda.is_bf16_supported(),
    bf16_full_eval=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    fp16_full_eval=not torch.cuda.is_bf16_supported(),
    tf32=True,
    # load_best_model_at_end=True,
    # metric_for_best_model='F1',
    # greater_is_better=True,
    optim='adamw_torch_fused',
    # resume_from_checkpoint=pretrained_model_path,
    report_to='wandb',
    run_name="siglip-large-augv2-10ep",
    gradient_checkpointing=False,
    torch_compile=sys.platform == 'linux',
)

# freeze vision or text backbone
# for param in model.vision_model.parameters(): param.requires_grad = False
# for param in model.text_model.parameters(): param.requires_grad = False

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)
train_result = trainer.train()
