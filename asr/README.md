# ASR

## Input

Audio file provided in `.WAV` format with a sample rate of 16 kHz.

https://github.com/TIL-24/til-24-base/assets/162278270/5e42363d-9f01-4626-8d70-cc7dc8e48c71

Note that the above example is in `mp4` format as GitHub does not support embedding `.wav` files in README files. However, audio files provided on GCP will be `.wav` files.

## Output

Transcription of audio file. Example: `"Heading is one five zero, target is green commercial aircraft, tool to deploy is electromagnetic pulse."`


# Submissions
### Whisper Small en non combined with niner:
- Accuracy: 0.9934860264761505
- Speed Score: 0.825095724074074

### Whisper Small en combined with niner:
- Accuracy: 0.9926455137633957
- Speed Score: 0.8088188985185185

### Whisper Small en combined without niner:
- Accuracy: 0.9922252574070183
- Speed Score: 0.8239869433333333

### Parakeet RNNT 0.6b non combined:
- Accuracy: 0.9686909014498845
- Speed Score: 0.8335885779629629

### Parakeet RNNT 0.6b combined:
- Accuracy: 0.9892834629123766
- Speed Score: 0.8456000411111111