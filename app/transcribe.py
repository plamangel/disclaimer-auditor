import os
import whisper

def transcribe_audio(file_path: str, language: str | None = None) -> str:
    model_name = os.environ.get("WHISPER_MODEL", "base")
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path, language=language)
    # Join segments to plain text. For MVP we ignore diarization.
    text = result.get("text", "").strip()
    return text
