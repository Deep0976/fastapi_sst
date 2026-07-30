from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import os

app = FastAPI()

HF_MODEL = os.environ.get("HF_MODEL", "openai/whisper-base")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_API_TOKEN = os.environ["HF_API_TOKEN"]

@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    audio_bytes = audio.file.read()

    try:
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            data=audio_bytes,
            timeout=60,
        )
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="Hugging Face API unreachable")

    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()
