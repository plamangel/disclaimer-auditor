import os
import sys
import csv
import json
from dotenv import load_dotenv

sys.path.append(os.path.abspath("."))
from app.policy_loader import load_policy
from app.llm_extractor import extract_flags
from app.scoring import aggregate_score
from app.utils import redact_pii
from sentence_transformers import SentenceTransformer

def main(paths):
    load_dotenv()
    policy_path = os.environ.get("POLICY_PATH", "policy/policy_disclaimer.yml")
    policy = load_policy(policy_path)
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    os.makedirs("outputs", exist_ok=True)
    index_csv = os.path.join("outputs", "index.csv")
    if not os.path.exists(index_csv):
        with open(index_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id","filename","score","verdict","transcript_chars"])

    for p in paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            continue

        text = open(p, "r", encoding="utf-8").read()
        redacted = redact_pii(text)
        flags = extract_flags(redacted, policy)
        total, verdict, breakdown, evidence_map = aggregate_score(redacted, flags, policy, embed_model)

        base = os.path.splitext(os.path.basename(p))[0]
        outj = os.path.join("outputs", f"{base}.json")
        with open(outj, "w", encoding="utf-8") as f:
            json.dump({
                "id": base,
                "filename": os.path.basename(p),
                "score": round(total, 3),
                "verdict": verdict,
                "breakdown": breakdown,
                "evidence": evidence_map,
                "transcript_chars": len(redacted),
                "model_info": {
                    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                    "use_llm_extractor": os.environ.get("USE_LLM_EXTRACTOR","false")
                }
            }, f, ensure_ascii=False, indent=2)

        with open(index_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([base, os.path.basename(p), round(total,3), verdict, len(redacted)])

        print(f"Done: {outj} | Verdict: {verdict} | Score: {round(total,3)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/process_transcript.py <file1.txt> [file2.txt ...]")
        sys.exit(1)
    main(sys.argv[1:])
