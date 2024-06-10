import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['HF_HOME'] = 'medium_model'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from faster_whisper import WhisperModel
MODEL_PATH= "medium_model"
# medium epoch5 0.99569 

class ASRManager:
    def __init__(self):
        # initialize the model here
        self.frequency = 16000
        self.model = WhisperModel(MODEL_PATH, device = "cuda", compute_type="float16", local_files_only = True)
        
    def clean(annotation):
        if "'" in annotation:
            # print(annotation, f'has \' in {annotation}, removing')
            annotation = annotation.split("'")[0] + annotation.split("'")[1][1:]  # Tokenizer includes "'" but TIL dataset does not, remove the S following '
        return annotation
        
    def batch_transcribe(self, batch) -> str:
        # batch is a list of audio waveforms
        # This is extremely rudimentary. Currently looking at possible batch inference using faster-whisper
        batchResponse = []
        for wf in batch:
            output = ""
            # segments, _ = self.model.transcribe(wf, beam_size = 5, vad_filter = True)
            segments, _ = self.model.transcribe(wf, beam_size = 5)
            for s in segments:
                output += s.text
            batchResponse.append(output)
            
        return batchResponse
    
    def batch_transcribe_vad(self, batch, vad_ms=500) -> str:
        batchResponse = []
        for wf in batch:
            output = ""
            segments, _ = self.model.transcribe(
                wf, 
                beam_size = 5, 
                vad_filter = True,
                vad_parameters=dict(
                    min_silence_duration_ms=vad_ms,
                    threshold=0.1
                )
            )
            # segments, _ = self.model.transcribe(wf, beam_size = 5)
            for s in segments:
                output += s.text
            batchResponse.append(output)
            
        return batchResponse

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        
        segments, _ = self.model.transcribe(w, beam_size=5)
        output = ""
        for segment in segments:
            output += segment.text
            
        return output