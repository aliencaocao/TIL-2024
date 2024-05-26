import os
os.environ['HF_HOME'] = 'huggingface'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = 'True'
import math
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import evaluate
from torch.utils.data import DataLoader



model_name = 'openai/whisper-small.en'
checkpoint_name = 'whisper-checkpoints/checkpoint-750/'
processor = WhisperProcessor.from_pretrained(model_name)

full_audio_path = "audio"
# Use audiofolder metadata to load the audio dataset
ds = load_dataset('audiofolder', data_dir=full_audio_path, split='train')

ds = ds.train_test_split(test_size=0.2, shuffle=True)

def prepare_dataset(batch):
    model_name = 'openai/whisper-small.en'

    # separate import for each process
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(model_name, resume_download= None)

    batch["input_features"] =[processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0] for audio in batch["audio"]]
    batch["input_length"] = [len(b) for b in batch["input_features"]]
    batch["labels"] = processor(text=batch["transcription"]).input_ids
    batch['length'] = batch["input_length"]

    return batch

print("Prep data?")
if __name__ == '__main__':
    print("Prepping dataset")
    ds = ds.map(prepare_dataset, num_proc=8, batched=True, batch_size=256)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = WhisperForConditionalGeneration.from_pretrained(
    model_name,  # checkpoint_name
    pad_token_id=processor.tokenizer.pad_token_id,
    mask_time_prob=0.5,  # 0.05
    mask_time_length=10, # 10
    mask_feature_prob=0.5, # 0
    mask_feature_length=10, # 10
    apply_spec_augment=True
)


model.freeze_encoder()

per_gpu_bs = 2
effective_bs = 32
training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-checkpoints",
    overwrite_output_dir =True,
    per_device_train_batch_size=per_gpu_bs,
    gradient_accumulation_steps=math.ceil(effective_bs/per_gpu_bs),
    learning_rate=1e-4,
    num_train_epochs=20,
    gradient_checkpointing=False,
    fp16=True,
    # bf16=True,  # for A100
    fp16_full_eval=True,
    # bf16_full_eval=True,  # for A100
    group_by_length=True,  # slows down
    evaluation_strategy="epoch",
    save_strategy='epoch',  # epoch
    save_safetensors=True,
    per_device_eval_batch_size=4,
    save_steps=1,
    eval_steps=1,
    logging_steps=100,
    save_total_limit=3,
    lr_scheduler_type='cosine',
    load_best_model_at_end=True,  # True
    adam_beta1=0.9,
    adam_beta2=0.98,  # follow fairseq fintuning config
    warmup_ratio=0.22, # follow Ranger21
    weight_decay=1e-4,  # follow Ranger21
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to=['tensorboard'],
    dataloader_num_workers=1 if os.name != 'nt' else 1)


class CustomWhisperTrainer(Seq2SeqTrainer):
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if os.name != 'nt':
            accelerator.backward(self.scaler.scale(loss))
            # self.scaler.scale(loss).backward()
        else:
            self.scaler.scale(loss).backward()
        return loss.detach()
    
if os.name != 'nt':
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision='fp16', dynamo_backend='eager')  # FP8 needs transformer_engine package which is only on Linux with Hopper GPUs

def tri_stage_schedule(epoch: int, max_epoch = training_args.num_train_epochs, stage_ratio = [0.1, 0.4, 0.5], peak_lr = training_args.learning_rate, initial_lr_scale=0.01, final_lr_scale=0.05):
    """https://github.com/facebookresearch/fairseq/blob/5ecbbf58d6e80b917340bcbf9d7bdbb539f0f92b/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py#L51"""
    assert sum(stage_ratio) == 1
    current_ratio = epoch / max_epoch
    if current_ratio < stage_ratio[0]:  # linear warmup
        lrs = torch.linspace(initial_lr_scale * peak_lr, peak_lr, int(stage_ratio[0] * max_epoch))
        return lrs[epoch]
    elif stage_ratio[0] <= current_ratio <= stage_ratio[1]:  # constant
        return peak_lr
    else:  # exponential decay
        decay_factor = -math.log(final_lr_scale) / (stage_ratio[2] * max_epoch)
        return peak_lr * math.exp(-decay_factor * stage_ratio[2] * max_epoch)
    
trainer = CustomWhisperTrainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # optimizers=(optimizer, scheduler),
)
if os.name != 'nt':  # windows does not support torch.compile yet
    trainer.model_wrapped, trainer.optimizer, trainer.lr_scheduler = accelerator.prepare(trainer.model_wrapped, trainer.optimizer, trainer.lr_scheduler)

trainer.train()
if os.name != 'nt':
    accelerator.wait_for_everyone()