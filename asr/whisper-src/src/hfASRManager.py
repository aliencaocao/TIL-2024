# from transformers import WhisperProcessor
# import os
# os.environ['HF_HOME'] = 'huggingface'
# os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
# os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = 'True'
# from transformers import pipeline
# from transformers.pipelines.pt_utils import KeyDataset

import os
os.environ['HF_HOME'] = 'huggingface'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = 'True'
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
weights = "model"

# print(np.array(waveform, dtype= float))



class ASRManager:
    def __init__(self):
        # initialize the model here
        self.pipe = pipeline(task="automatic-speech-recognition", model=weights, device=0)
        
        
    def batch_transcribe(self, batch) -> str:
        # batch is a list of audio waveforms
        # This is extremely rudimentary. Currently looking at possible batch inference using faster-whisper
            
        return [x['text'] for x in self.pipe(batch)]
        
