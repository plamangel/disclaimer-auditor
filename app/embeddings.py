from typing import Optional
import os
from sentence_transformers import SentenceTransformer

_embed_model: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    """
    Singleton SentenceTransformer used across the app.
    Loads once per process to avoid repeated heavy inits.
    """
    global _embed_model
    if _embed_model is None:
        # Limit threading for deterministic results
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        # Keep the model consistent with policy_loader + scoring
        _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embed_model