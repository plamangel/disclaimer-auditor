import os
import json
from typing import Dict, Any, Optional
import requests

def _ollama_extract(text: str, policy: Dict[str, Any], model_name: str) -> Optional[Dict[str, str]]:
    prompt = _build_prompt(text, policy)
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        out = r.json().get("response", "").strip()
        return _safe_json(out)
    except Exception:
        return None

def _openai_extract(text: str, policy: Dict[str, Any]) -> Optional[Dict[str, str]]:
    from openai import OpenAI
    client = OpenAI()
    prompt = _build_prompt(text, policy)
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a strict compliance extractor that only returns VALID JSON and nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        out = resp.choices[0].message.content.strip()
        return _safe_json(out)
    except Exception:
        return None

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
    ids = ", ".join([r["id"] for r in policy["requirements"]])
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
Your task: for EACH requirement ID, decide if the agent explicitly communicated it, even if paraphrased.

Return ONLY valid JSON with keys EXACTLY the requirement IDs from the policy, and values from this set:
"true" | "false" | "unclear".

Polarity and contradictions matter. Apply these strict rules:

1) no_miles_accrual:
   - If the agent says the ticket is NOT eligible to earn/collect/accrue miles or points, or miles will NOT be credited → "true".
   - If the agent says or implies the ticket IS eligible to earn/collect/accrue miles or points, or miles WILL be credited → "false".
   - Mentions like "no bonuses/promos/airline discounts apply" count toward "true" when they are clearly tied to miles/points eligibility.
   - If not clearly stated either way → "unclear".

2) corporate_miles_discount:
   - If the agent explains the ticket is non-fare/discounted because it is issued with corporate miles/points, coupons, or vouchers → "true".
   - If the agent attributes discounting to something else unrelated (and does NOT mention corporate miles/points/coupons/vouchers) → "unclear".
   - This rule is independent of miles accrual; do not infer one from the other.

3) no_personal_info:
   - If the agent tells the customer NOT to link/add any personal info to the booking (email, phone, frequent flyer) → "true".
   - Phrases like "please do not link any personal information" or warnings such as "an attempt to link personal frequent flyer/email/phone may cause suspension" → "true".
   - If the agent allows or suggests linking/adding personal info → "false".
   - Else → "unclear".

4) keep_agency_contacts:
   - If the agent states the booking is linked to the agency’s email/contacts and these MUST remain unchanged → "true".
   - If the agent allows or suggests changing/replacing those contacts with personal ones → "false".
   - Else → "unclear".

5) contact_us_changes_24_7:
   - If the agent instructs the customer to contact the agent directly (e.g., "for any changes or cancellations or any other requests please call/reach me directly") and/or the agency’s customer support (24/7) → "true".
   - If the agent directs the customer to contact the airline for changes → "false".
   - Else → "unclear".

6) do_not_contact_airline:
   - If the agent explicitly prohibits contacting/reaching/calling the airline/carrier → "true".
   - If the agent says or implies the customer MAY contact the airline/carrier, or suggests doing so → "false".
   - If nothing is said about contacting the airline → "unclear".

General principles:
- Be conservative: if a requirement is not explicit, return "unclear".
- If any explicit contradiction exists, prefer "false" over "true".
- Do NOT invent information. Base your judgment ONLY on the provided transcript.
- Output ONLY JSON. No prose, no markdown, no explanations.

Now, return a JSON object with keys equal to these requirement IDs (exactly):
{ids}

Return JSON like:
{example_json}

AGENT transcript:
{agent_text}
"""
    return header.strip()

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
    return {r["id"]: "unclear" for r in policy["requirements"]}