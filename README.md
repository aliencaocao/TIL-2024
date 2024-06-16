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
| ASR  | Whisper Medium En                                     | 0.9956923723471317 |
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

| Model                | Train Set                                             | Leaderboard Score |
|----------------------|-------------------------------------------------------|-------------------|
| Whisper Small        | DSTA set                                              | 0.9935            |
| Whisper Small        | DSTA set (Augs)                                       | 0.9940            |
| Whisper Small        | DSTA set + XS set                                     | 0.9922            |
| Whisper Small (niner)| DSTA set + XS set                                     | 0.9926            |

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

#### Model Conversion
```python
from faster_whisper import WhisperModel
from ctranslate2.converters import TransformersConverter

processor = WhisperProcessor.from_pretrained('<your-whisper-model>',resume_download = None)
processor.tokenizer.save_pretrained(weights)
processor.feature_extractor.save_pretrained(weights)

outputDir = "<output-directory>"
ModelConverter = TransformersConverter(weights)
modelPath = ModelConverter.convert(
    outputDir,
    force = True
)
```

#### Running inference using faster-whisper

```python
import librosa
filename = "<your-wav-file>" 

frequency = 16000
waveform, sr = librosa.load(filename, sr = frequency)

weights = "<your-model-weights>"
model = WhisperModel(weights, device = "cuda", compute_type="float16")

segments, info = model.transcribe(waveform, beam_size=5)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

## NLP
### Data Augmentation

### Model

### Training

### Inference


## VLM
### Data Augmentation

### Model

### Training

### Inference
