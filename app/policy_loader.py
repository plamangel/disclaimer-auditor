import yaml
from typing import Any, Dict

def load_policy(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)
