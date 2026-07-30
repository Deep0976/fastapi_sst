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
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_API_TOKEN = os.environ["HF_API_TOKEN"]

NORMALIZE_PROMPT = (
    "You convert Hindi/Urdu text into Hindi written in Devanagari script. "
    "Do not translate the meaning or language, only convert the script if needed. "
    "Output only the converted text, nothing else."
)

def normalize_to_hindi(text: str) -> str:
    for attempt in range(2):
        try:
            response = requests.post(
                HF_CHAT_URL,
                headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
                json={
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "messages": [
                        {"role": "system", "content": NORMALIZE_PROMPT},
                        {"role": "user", "content": text},
                    ],
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException:
            if attempt == 1:
                return text

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
            timeout=300,
        )
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="Hugging Face API unreachable")

    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    result = response.json()
    if "text" in result:
        result["text"] = normalize_to_hindi(result["text"])

    return result
