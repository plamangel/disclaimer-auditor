from typing import Optional
from sentence_transformers import SentenceTransformer

_embed_model: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    """
    Singleton SentenceTransformer used across the app.
    Loads once per process to avoid repeated heavy inits.
    """
    global _embed_model
    if _embed_model is None:
        # Keep the model consistent with policy_loader + scoring
        _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embed_model