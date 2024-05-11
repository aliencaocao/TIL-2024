from fastapi import FastAPI, Request
import base64
from ASRManager import ASRManager

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

    # get base64 encoded string of audio, convert back into bytes
    input_json = await request.json()

    predictions = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        audio_bytes = base64.b64decode(instance["b64"])

        transcription = asr_manager.transcribe(audio_bytes)
        predictions.append(transcription)

    return {"predictions": predictions}
