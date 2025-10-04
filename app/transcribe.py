# app/transcribe.py
import os
from typing import Any, Dict, List, Optional
from faster_whisper import WhisperModel
import ctranslate2

_model = None

def _pick_compute_type(device: str) -> str:
    """
    Choose a supported compute_type for the current device.
    Preference order: int8_float16 -> int8 -> float16 -> float32.
    Falls back safely if not supported.
    """
    requested = os.environ.get("WHISPER_COMPUTE", "").strip()  # optional manual override
    supported = set(ctranslate2.get_supported_compute_types(device or "cpu"))

    if requested:
        if requested in supported:
            return requested
        # if unsupported, continue to auto-pick

    for ct in ("int8_float16", "int8", "float16", "float32"):
        if ct in supported:
            return ct
    return "float32"

def _get_model():
    global _model
    if _model is None:
        model_name = os.environ.get("WHISPER_MODEL", "small")
        device = os.environ.get("WHISPER_DEVICE", "cpu").lower()
        compute_type = _pick_compute_type(device)
        _model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _model

def transcribe_audio(file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe an audio file with faster-whisper and return:
      {
        "transcript": str,
        "segments": [{"text": str, "start": float, "end": float}, ...],
        "model_info": {"model": str, "device": str, "compute_type": str}
      }
    """
    model = _get_model()
    segments, info = model.transcribe(
        file_path,
        language=language,
        vad_filter=True,
        # Deterministic settings
        beam_size=1,                 # no alternative beams
        best_of=1,                   # disable multi-sample search
        temperature=0.0,             # no sampling randomness
        compression_ratio_threshold=None,  # avoid heuristic early-exit variability
    )

    segs: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        segs.append({
            "text": text,
            "start": float(s.start),
            "end": float(s.end),
        })
        full_text_parts.append(text)

    full_text = " ".join(full_text_parts).strip()

    device = os.environ.get("WHISPER_DEVICE", "cpu").lower()
    compute_type = _pick_compute_type(device)

    return {
        "transcript": full_text,
        "segments": segs,
        "model_info": {
            "model": os.environ.get("WHISPER_MODEL", "small"),
            "device": device,
            "compute_type": compute_type,
        },
    }