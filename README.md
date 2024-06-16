# DSTA BrainHack TIL-AI 2024

## Team 12000SGDPLUSHIE

## Introduction
TIL-AI 2024 has 3 tasks: Automatic Speech Recognition (ASR), Natural Language Processing (NLP) and Visual Language Modeling (VLM).

ASR task is to convert radio-distorted and noisy audio into text. NLP task is to convert text into structured data for controlling a turrent. VLM is to locate a flying object in an image based on its description. The three tasks chains up in Finals to drive a DJI Robomaster's turrent in an simulated environment.

## Team Members (in alphabetical order)
* [Billy Cao](https://github.com/aliencaocao) (L): NLP/VLM
* [Ho Wing Yip](https://github.com/HoWingYip): VLM
* [Huang Qirui](https://github.com/hqrui): ASR
* [Marcus Wee](https://github.com/Marcushadow): ASR
* [Ooi Xuan Shan](https://github.com/ooixs): VLM

## Achievements
* 3rd overall in Qualifiers
* 2nd in Semi-finals
* 7th in Finals

Unfortunately our model may have overfitted to leaderboard hidden test set in Finals, resulting in a lower ranking.

## Final evaluation results on leaderboard
| Task | Model                                                 | Accuracy Score     |
|------|-------------------------------------------------------|--------------------|
| ASR  | Whisper Medium                                        | 0.9956923723471317 |
| NLP  | gorilla-openfunctions-v2                              | 0.99933333         |
| VLM  | YOLOv6l6 + RealESRGAN-x4v3 + SigLIP-large-patch16-384 | 0.913              |

We do not report speed score here as it is not optimal in leaderboard submission since we employed hardware-specific optimizations. More details will be below.

## ASR
### Data Augmentation

For ASR data augmentations, we decided to use the `audiomentations` library. We do note that the `torch-audiomentations` is also an alternative that better utilises the GPU. Nonetheless, the `audiomentations` library still offered us a wider variety of augmentations, allowing us to train our ASR models to be more robust to different types of noise.

Below is a set of augmentations that worked well for us initially:

```python
augment = Compose([
    HighShelfFilter(max_gain_db=6.0, p=0.3),
    LowShelfFilter(max_gain_db=6.0, p=0.3),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2),
    BandPassFilter(p=0.3)
])
```

#### [High Shelf Filter](https://iver56.github.io/audiomentations/waveform_transforms/high_shelf_filter/)

Attenuates or boosts the higher end of frequencies. We decided to boost the effect of higher frequencies by up to 6.0 decibels in our training data.

#### [Low Shelf Filter](https://iver56.github.io/audiomentations/waveform_transforms/low_shelf_filter/)

Attenuates or boosts the lower end of frequencies. We decided to boost the effect of lower frequencies by up to 6.0 decibels in our training data

#### [Time Stretch](https://iver56.github.io/audiomentations/waveform_transforms/time_stretch/)

Change the speed or duration of the signal without changing the pitch. We decided to limit the rate between 0.9 and 1.1 to avoid distortion of the audio clips beyond human recognition.

#### [Band Pass Filter](https://iver56.github.io/audiomentations/waveform_transforms/band_pass_filter/)

Attenuates the low and high frequencies of an audio clip, which can help to simulate radio transmission which only allows certain frequencies to be broadcasted.

### Model

| Model                 | Train Set                | Leaderboard Score |
|-----------------------|--------------------------|-------------------|
| Whisper Small         | DSTA set                 | 0.9935            |
| Whisper Small         | DSTA set (Augs)          | 0.9940            |
| Whisper Small         | DSTA set + XS set        | 0.9922            |
| Whisper Small (niner) | DSTA set + XS set        | 0.9926            |
| Whisper Medium        | DSTA set (Augs)          | 0.9957            |
| Parakeet RNNT 0.6B    | DSTA set (Augs)          | 0.9687            |
| Parakeet RNNT 0.6B    | DSTA set + XS set (Augs) | 0.9893            |

Parakeet RNNT 0.6B gave a much worse leaderboard score despite a ~10x lower validation word error rate during training. Perhaps, Whisper has supreme robustness due to being trained on 680k hours of labelled data versus Parakeet's 64k hours.

### Training

We used the following configuration to train our whisper models: 

```python
training_args = Seq2SeqTrainingArguments(
    output_dir="<output_directory>",
    overwrite_output_dir =True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5, # was 1e-4
    warmup_steps=500,
    num_train_epochs=30,
    gradient_checkpointing=True,
    fp16=True,
    torch_compile=True,
    fp16_full_eval=True,
    evaluation_strategy="epoch",
    save_strategy='epoch',
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit = 3,
    adam_beta1=0.9,
    adam_beta2=0.98,  # follow fairseq finetuning config
    warmup_ratio=0.22, # follow Ranger21
    weight_decay=1e-4,
    dataloader_num_workers=psutil.cpu_count(logical=True)
)
```

#### Notable Configs

`warmup_steps`\
By gradually increasing the learning rate at the beginning of training, warmup steps help to prevent instability and divergence, leading to more efficient and effective training of neural network models.

`torch_compile=True`\
Which makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels. While this may

`dataloader_num_workers=psutil.cpu_count(logical=True)`\
As mentioned in this [blog](https://discuss.huggingface.co/t/help-needed-with-issues-while-trying-fine-tune-whisper/40248/2), optimising the number of dataloaders helps to feed the GPU more effectively.


### Inference
To speed up inference using whisper models, we decided to use [faster-whisper](https://github.com/SYSTRAN/faster-whisper), which utilises the ctranslate2, a fast inference engine for Transformer models, to increase the inference speeds by more than 2x.

## NLP
The task is to convert instructions on operating a turret into structured data with the following fields: heading (a 3-digit number), tool (the weapon to use), target (description of the target's color and type).

We approached the task as a function calling task common in LLM agents. We are aware of much smaller scale/faster pipelines available but decided to go the most time saving way of "whacking" a large model to solve everything so we can focus on the hard VLM task.
### Data Transformation
System prompt: `Convert the given instruction to turret into a function call to the control_turret function. Strictly ONLY one call is needed. If there are multiple instructions, use the FIRST one only.`

Function calling task require a function definition to be provided to the LLM, in JSON string and appended to the system prompt. After tuning by evaluating zero-shot on training data, ours is:
```json
{'name': 'control_turret',
'description': 'Control the turret by giving it heading, tool to use and target description',
'parameters': {'type': 'object', 'properties': {'heading':
{'type': 'string', 'description': 'Heading of target in three arabic numbers and multiples of five (005 to 360). Give None if not specified.'},
'tool': {'type': 'string', 'description': 'Tool to use or deploy. Give None if not specified.'},
'target': {'type': 'string', 'description': "Description of the target or enemy, exclude any quantifiers like 'the' or 'a'. It is a phrase that describe the appearance of the target like its color and type. Include ONLY the appearance and NOTHING else like its heading. Give None if not specified."}},
'required': ['heading', 'tool', 'target']}}
```

As sometimes the command can contain reputations and model can be confused, we did the following:
1. Check for “repeat” in text and attempt to extract information on first half, retry on full text if failed to extract all 3 fields. When retrying, omit the `Give None if not specified.` part of the prompt.
2. Any matched tool will be included in prompt so model do not choose it again as target (e.g. missiles)

To further reduce inference time, simple regex-based rules were used:
1. Regex to match 3-word sequence of heading e.g. "One One Zero"
2. Regex to match 7 known weapons in training data: `{'anti-air artillery', 'drone catcher', 'electromagnetic pulse', 'emp', 'interceptor jets', 'machine gun', 'surface-to-air missiles'}`

### Model
[Gorilla-OpenFunctions-v2](https://gorilla.cs.berkeley.edu/blogs/7_open_functions_v2.html) by UC Berkely, SOTA open-source model on [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), fine-tuned from LLaMA2-7B.

Using the above data transformations, the pretrained model scored 0.9978597402597402 on leaderboard.

### Training
We then fine-tuned the model using [Unsloth](https://github.com/unslothai/unsloth). Training took 34 minutes on Colab's free T4 GPU. We used [Rank-Stabilized](https://arxiv.org/abs/2312.03732) [LoRA (Low-Rank Adaptation of Large Language Models)](https://arxiv.org/abs/2106.09685).
#### Hyperparameters:
* LoRA rank: 16
* LoRA alpha: 16
* optimizer: AdamW 8bit
* initial LR: 2e-4
* warmup: 10 steps
* LR schedule: cosine annealing decay
* effective batchsize: 8
* epoch: 1
* weight decay: 0.01

Training code can be found in [train.ipynb](nlp/src/train.ipynb).

### Inference
LLMs may seem extreme for this task for its sheer size and inference cost, but it CAN be deployed efficiently with inference costs matchihng smaller models like BERT.

We used [ExLlamaV2](https://github.com/turboderp/exllamav2) quantization and inference engine. It was chosen because it does all matrix ops in FP16 which Tesla T4 GPU in submission runtime excels at. It is also specifically optimized for consumer Nvidia GPU which is used in the Finals (RTX 4070Ti Super).

Our model was quantized to average of 5.0 bit with 6-bit head precision, using default calibration data. Anything below 5.0 bit severely reduces model performance even with custom calibration data.

We made tweaks to further reduce VRAM I/O bottlenecks to speed up inference:
* Discard logits in VRAM instead of copying back to CPU (2x+ speedup), controlled by `config.max_output_len = 1`
* restricting KV cache pool to short context length (512) and batch size (4 in qualifiers, 1 in finals), controlled by `config.max_seq_len = 512` and `config.max_batch_size = 4`

Our trained model has 0.9993333333333333 leaderboard score without any tricks / regex mentioned above, just to test fine-tuning effectiveness. In finals, we still employ those tricks.

Model peak VRAM usage: 5.7GB

Time taken to process 1 sample at bs=1: 0.36s. With bs=4, it is about 2 times faster per sample.

Inference code can be found in [NLPManager.py](nlp/src/NLPManager.py).


## VLM
### Data Augmentation

### Model

### Training

### Inference
