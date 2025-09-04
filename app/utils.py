import re
from typing import List

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
