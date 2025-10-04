import os
from typing import Any, Dict, List, Optional
import whisper

def transcribe_audio(file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe an audio file with Whisper and return:
      {
        "transcript": str,
        "segments": [{"text": str, "start": float, "end": float}, ...],
        "model_info": {"model": str, "device": str, "fp16": bool}
      }
    """
    model_name = os.environ.get("WHISPER_MODEL", "base")
    model = whisper.load_model(model_name)

    # Use FP32 on CPU (avoids the usual warning and speeds up a bit on Macs without GPU)
    use_fp16 = (getattr(model, "device", None) is not None and str(model.device).startswith("cuda"))
    result = model.transcribe(
        file_path,
        language=language,
        verbose=False,
        fp16=use_fp16,
        # word_timestamps=False  # sentence-level timestamps are enough for our use-case
    )

    segments: List[Dict[str, Any]] = []
    for seg in (result.get("segments") or []):
        segments.append({
            "text": (seg.get("text") or "").strip(),
            "start": float(seg.get("start", 0.0)),  # seconds
            "end": float(seg.get("end", 0.0)),      # seconds
        })

    full_text = (result.get("text") or "").strip()

    out: Dict[str, Any] = {
        "transcript": full_text,
        "segments": segments,
        "model_info": {
            "model": model_name,
            "device": str(getattr(model, "device", "cpu")),
            "fp16": use_fp16,
        },
    }
    return out