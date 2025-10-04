import os
import json
from typing import Dict, Any, Optional
import requests

# -------- Speed/robustness knobs --------
MAX_TRANSCRIPT_CHARS = int(os.environ.get("LLM_MAX_TRANSCRIPT_CHARS", "4000"))
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "60"))
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _truncate_center(text: str, max_len: int) -> str:
    """Keep head+tail to preserve intro + conclusion; speeds up LLM."""
    if not text or len(text) <= max_len:
        return text or ""
    head = max_len * 2 // 3  # ~66%
    tail = max_len - head
    return text[:head] + "\n...\n" + text[-tail:]


def _compact_ids(policy: Dict[str, Any]) -> str:
    return ", ".join([r["id"] for r in policy.get("requirements", [])])


def _safe_json(s: str) -> Optional[Dict[str, str]]:
    try:
        return json.loads(s)
    except Exception:
        # try to extract JSON block heuristically
        import re
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def _build_prompt(agent_text: str, policy: Dict[str, Any]) -> str:
    ids = _compact_ids(policy)
    example_json = (
        '{\n'
        '  "no_miles_accrual": "true|false|unclear",\n'
        '  "corporate_miles_discount": "true|false|unclear",\n'
        '  "no_personal_info": "true|false|unclear",\n'
        '  "keep_agency_contacts": "true|false|unclear",\n'
        '  "contact_us_changes_24_7": "true|false|unclear",\n'
        '  "do_not_contact_airline": "true|false|unclear"\n'
        '}'
    )
    header = f"""
You will receive the AGENT-only transcript from a call and a set of policy requirement IDs.
Task: for EACH requirement ID, decide if the agent explicitly communicated it, even if paraphrased.

Return ONLY valid JSON with keys EXACTLY the requirement IDs and values in: "true" | "false" | "unclear".
Rules:
- Be conservative: if not explicit, return "unclear".
- If any explicit contradiction exists, prefer "false" over "true".
- Handle polarity for miles: "eligible/earn/credited" contradicts "not eligible/won't accrue".
- Do NOT invent information. Base judgment ONLY on the transcript.
- Output ONLY JSON. No prose, no markdown.

IDs:
{ids}

Return JSON like:
{example_json}

AGENT transcript:
""".strip()
    return header + "\n" + agent_text


# ----------------- Ollama path (local, fast) -----------------

def _ollama_extract(text: str, policy: Dict[str, Any], model_name: str) -> Optional[Dict[str, str]]:
    prompt = _build_prompt(_truncate_center(text, MAX_TRANSCRIPT_CHARS), policy)
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                # speed/consistency options
                "options": {
                    "num_predict": 160,      # small, enough for 6 keys
                    "temperature": 0,
                    "num_ctx": 2048,
                    "stop": ["\n\nEND"],
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        out = r.json().get("response", "").strip()
        return _safe_json(out)
    except Exception:
        return None


# ----------------- OpenAI path (fallback) -----------------

def _openai_extract(text: str, policy: Dict[str, Any]) -> Optional[Dict[str, str]]:
    from openai import OpenAI
    client = OpenAI(timeout=OPENAI_TIMEOUT)
    prompt = _build_prompt(_truncate_center(text, MAX_TRANSCRIPT_CHARS), policy)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_DEFAULT,
            messages=[
                {"role": "system", "content": "You are a strict compliance extractor that only returns VALID JSON and nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=220,
        )
        out = (resp.choices[0].message.content or "").strip()
        js = _safe_json(out)
        if isinstance(js, dict):
            return js
        return None
    except Exception:
        return None


def extract_flags(agent_text: str, policy: Dict[str, Any]) -> Dict[str, str]:
    if os.environ.get("USE_LLM_EXTRACTOR", "false").lower() == "true":
        # Ollama first if configured
        model_name = os.environ.get("OLLAMA_MODEL", "").strip()
        if model_name:
            out = _ollama_extract(agent_text, policy, model_name)
            if isinstance(out, dict):
                return out

        # OpenAI fallback if API key is present
        if os.environ.get("OPENAI_API_KEY"):
            out = _openai_extract(agent_text, policy)
            if isinstance(out, dict):
                return out

    # default: mark all as 'unclear'
    return {r["id"]: "unclear" for r in policy.get("requirements", [])}