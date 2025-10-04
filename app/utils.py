import re
from typing import List, Tuple, Dict, Any

SENT_SPLIT_RE = re.compile(r'(?<=[\.!\?])\s+')

def split_sentences_with_times(segments: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Split transcript into sentences, carrying over segment start/end (ms).
    Each sentence inherits its parent segment's [start_ms, end_ms].

    Returns:
      sentences: List[str]
      times:     List[Tuple[start_ms, end_ms]] aligned 1:1 with sentences
    """
    sentences: List[str] = []
    times: List[Tuple[int, int]] = []
    for seg in segments or []:
        text = (seg.get("text") or "").strip()
        start_ms = int(float(seg.get("start", 0)) * 1000)
        end_ms = int(float(seg.get("end", 0)) * 1000)
        if not text:
            continue
        parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]
        if not parts:
            # fallback: use whole segment as one sentence
            sentences.append(text)
            times.append((start_ms, end_ms))
            continue
        for p in parts:
            sentences.append(p)
            times.append((start_ms, end_ms))
    return sentences, times

def format_mmss(ms: int) -> str:
    s = max(ms, 0) // 1000
    return f"{s//60:02d}:{s%60:02d}"

PII_PATTERNS = [
    r'\b[0-9]{13,16}\b',  # potential card numbers
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',  # emails
    r'\+?[0-9][0-9\-\s]{6,}[0-9]',  # phone-like
]

def redact_pii(text: str) -> str:
    red = text
    for pat in PII_PATTERNS:
        red = re.sub(pat, "[REDACTED]", red)
    return red

def split_sentences(text: str) -> List[str]:
    # naive splitter; consider spacy for higher quality
    parts = re.split(r'[.!?\n]+', text)
    return [p.strip() for p in parts if p.strip()]