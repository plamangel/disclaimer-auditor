
import yaml
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer

# Singleton embedding model (loaded once per process)
_embed_model = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embed_model


def _precompute_example_embeddings(policy: Dict[str, Any]) -> None:
    """Attach precomputed embeddings for each requirement's canonical_examples.
    Stored under key "_ex_emb" as a list (JSON-serializable). No-op if empty.
    """
    model = _get_embed_model()
    for req in policy.get("requirements", []):
        examples: List[str] = req.get("canonical_examples") or []
        if examples:
            ex_emb = model.encode(examples, normalize_embeddings=True)
            # store as Python lists to keep the policy JSON/YAML serializable if needed
            req["_ex_emb"] = ex_emb.tolist()
        else:
            req["_ex_emb"] = []


def load_policy(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        policy: Dict[str, Any] = yaml.safe_load(f)

    # Precompute canonical example embeddings once (used by scoring for speed)
    _precompute_example_embeddings(policy)

    return policy
