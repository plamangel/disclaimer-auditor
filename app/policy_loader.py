import yaml
from typing import Any, Dict, List
from .embeddings import get_embed_model

def _precompute_example_embeddings(policy: Dict[str, Any]) -> None:
    """
    Attach precomputed embeddings for each requirement's canonical_examples.
    Stored under key "_ex_emb" as lists (JSON-serializable).
    """
    model = get_embed_model()
    for req in policy.get("requirements", []):
        examples: List[str] = req.get("canonical_examples") or []
        if examples:
            ex_emb = model.encode(examples, normalize_embeddings=True)
            req["_ex_emb"] = ex_emb.tolist()
        else:
            req["_ex_emb"] = []

def load_policy(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        policy: Dict[str, Any] = yaml.safe_load(f)

    _precompute_example_embeddings(policy)
    return policy