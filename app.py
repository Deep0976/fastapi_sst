from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

HF_MODEL = os.environ.get("HF_MODEL", "openai/whisper-base")
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
HF_API_TOKEN = os.environ["HF_API_TOKEN"]

@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    audio_bytes = audio.file.read()

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    if audio.content_type:
        headers["Content-Type"] = audio.content_type

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=audio_bytes,
            timeout=60,
        )
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="Hugging Face API unreachable")

    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()
