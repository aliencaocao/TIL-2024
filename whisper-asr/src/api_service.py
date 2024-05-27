from fastapi import FastAPI, Request
import base64
import librosa
from ASRManager import ASRManager
# from nfASRManager import ASRManager
import io
import pyloudnorm as pyln
SR = 16000

app = FastAPI()

asr_manager = ASRManager()

@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/stt")
async def stt(request: Request):
    """
    Performs ASR given the filepath of an audio file
    Returns transcription of the audio
    """

    input_json = await request.json()
    wflist = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        audio_bytes = base64.b64decode(instance["b64"])
        w, _ = librosa.load(io.BytesIO(audio_bytes), sr = SR)
        
        #  Loudness normalisation
        pna = pyln.normalize.peak(w, 1.0)
        meter = pyln.Meter(SR) # create BS.1770 meter
        loudness = meter.integrated_loudness(pna)
        pna = pyln.normalize.loudness(pna, loudness, 0.0)
        
        wflist.append(pna)
        
    # predictions = asr_manager.batch_transcribe(wflist)
    predictions = asr_manager.batch_transcribe_vad(wflist)

    return {"predictions": predictions}
