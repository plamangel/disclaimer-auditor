import hashlib
import json
import os
from typing import Any, Dict, Optional

CACHE_DIR = os.environ.get("LLM_CACHE_DIR", ".cache")

def _ensure_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

def make_key(text: str, policy_name: str) -> str:
    h = hashlib.sha256((text + "::" + (policy_name or "")).encode("utf-8")).hexdigest()
    return h

def get_cached(key: str) -> Optional[Dict[str, Any]]:
    _ensure_dir()
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def put_cached(key: str, obj: Dict[str, Any]) -> None:
    _ensure_dir()
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except Exception:
        pass