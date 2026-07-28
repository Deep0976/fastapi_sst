from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
import os

app = FastAPI()

print("Loading Whisper model...")
#model = whisper.load_model("base")
model = whisper.load_model("tiny")
print("Whisper model loaded!")

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(await audio.read())
        temp_path = temp_audio.name

    result = model.transcribe(
    temp_path,
    language="hi"
)

    os.remove(temp_path)

    return {
    "text": result["text"],
    "language": result["language"]
}
