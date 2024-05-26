import torch
from datasets import load_dataset
from PIL import Image
import sys
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoImageProcessor,
    AutoProcessor,
    EvalPrediction
)
from modeling_siglip import SiglipModel

pretrained_model_path = "../siglip-large-patch16-384"

model = SiglipModel.from_pretrained(pretrained_model_path).to('cuda')
processor = AutoProcessor.from_pretrained(pretrained_model_path)
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
# image_processor = AutoImageProcessor.from_pretrained(pretrained_model_path)
config = model.config

# split the ds into train and val. the image column is in format "image_1234_1.jpg". Take 1234 as the image id. Check if the id is smaller than 4086. If yes, put this in train, else in val.
def get_ds():
    dataset = load_dataset(path='.', data_files="til_siglip_ds.json")
    train_dataset = dataset.filter(lambda example: int(example["image"].split("_")[1]) < 4086)
    val_dataset = dataset.filter(lambda example: int(example["image"].split("_")[1]) >= 4086)
    return train_dataset, val_dataset

train_dataset, val_dataset = get_ds()

class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.NEAREST),  # lesson from TIL 2023: https://github.com/aliencaocao/TIL-2023?tab=readme-ov-file#finals-specific-tuning-2
            CenterCrop(image_size),
            ConvertImageDtype(torch.float16),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x

# For preprocessing the datasets.
# Initialize torchvision transforms and jit it for faster processing.
image_transformations = Transform(
    config.vision_config.image_size, processor.image_processor.image_mean, processor.image_processor.image_std
)
image_transformations = torch.jit.script(image_transformations)


def preprocess_dataset(dataset):
    # Preprocessing the datasets.
    data = dataset['train']
    # We need to tokenize inputs and targets.
    column_names = data.column_names

    # 6. Get the column names for input/target.
    image_column = "image"
    caption_column = "label"
    dataset_columns = (image_column, caption_column)

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
        image_dir = 'til_siglip_ds/'
        images = [read_image(image_dir + image_file, mode=ImageReadMode.RGB) for image_file in examples[image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    # Transform images on the fly as doing it on the whole dataset takes too much time.
    data.set_transform(transform_images)
    return data

train_dataset = preprocess_dataset(train_dataset)
val_dataset = preprocess_dataset(val_dataset)

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


# initialize Trainer
training_args = TrainingArguments(
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=1e-4,
    per_device_train_batch_size=4,
    remove_unused_columns=False,
    output_dir="siglip-finetune",
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    adam_beta1=0.9,
    adam_beta2=0.99,  # decrease from 0.999
    num_train_epochs=20,
    lr_scheduler_type="cosine",
    logging_strategy="epoch",
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=5,
    bf16=torch.cuda.is_bf16_supported(),
    bf16_full_eval=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    fp16_full_eval=not torch.cuda.is_bf16_supported(),
    tf32=True,
    dataloader_num_workers=4 if sys.platform == 'linux' else 0,  # no work on windows
    load_best_model_at_end=True,
    metric_for_best_model='F1',
    greater_is_better=True,
    # optim='adamw_torch_fused',
    optim='adafactor',
    # resume_from_checkpoint=False,
    report_to='none',
    gradient_checkpointing=True,
    torch_compile = sys.platform == 'linux',
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)
train_result = trainer.train()