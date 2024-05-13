from fastapi import FastAPI, Request

from NLPManager import NLPManager

app = FastAPI()

nlp_manager = NLPManager('models/gorilla-openfunctions-v2-5.0bpw-h6-exl2')


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/extract")
async def extract(instance: Request):
    """
    Performs QA extraction given a context string

    returns a dictionary with fields:

    {
        "heading": str,
        "target": str,
        "tool": str,
    }
    """
    # get transcription, and pass to NLP model
    request_dict = await instance.json()

    predictions = []
    transcripts = [instance["transcript"] for instance in request_dict["instances"]]
    # each is a dict with one key "transcript" and the transcription as a string
    answers = nlp_manager.qa(transcripts)
    predictions.extend(answers)

    return {"predictions": predictions}
