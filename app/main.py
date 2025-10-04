import os
import csv
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from dotenv import load_dotenv

from .policy_loader import load_policy
from .transcribe import transcribe_audio
from .llm_extractor import extract_flags
from .scoring import aggregate_score
from .utils import redact_pii, split_sentences_with_times
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="Disclaimer Auditor")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

POLICY_PATH = os.environ.get("POLICY_PATH", "policy/policy_disclaimer.yml")
policy = load_policy(POLICY_PATH)
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
INDEX_CSV = os.path.join(OUTPUT_DIR, "index.csv")
if not os.path.exists(INDEX_CSV):
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","filename","score","verdict","transcript_chars"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), language: str | None = Form(None)):
    # Save temp file
    fid = str(uuid.uuid4())
    temp_path = os.path.join(OUTPUT_DIR, f"{fid}_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Transcribe (returns dict with transcript + segments)
    tr = transcribe_audio(temp_path, language=language)
    transcript_text = tr.get("transcript", "")
    segments = tr.get("segments", [])  # list of {text, start, end} (seconds)

    # Redact PII in transcript and segments
    redacted_text = redact_pii(transcript_text)
    redacted_segments = []
    for seg in segments:
        redacted_segments.append({
            "text": redact_pii(seg.get("text", "")),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
        })

    # Build sentence-time bundle (ms) for scoring
    sentences, times = split_sentences_with_times(redacted_segments)

    # Inject bundle into scoring config (without mutating the loaded policy)
    scoring_cfg = dict(policy.get("scoring", {}))
    scoring_cfg["_sentence_times_bundle"] = {"sentences": sentences, "times": times}
    policy_for_scoring = {"requirements": policy["requirements"], "scoring": scoring_cfg}

    # Extract flags via LLM (optional) using redacted text only
    flags = extract_flags(redacted_text, policy)

    # Score using sentence-time bundle
    total, verdict, breakdown, evidence_map = aggregate_score(redacted_text, flags, policy_for_scoring, embed_model)

    # Persist JSON result
    result = {
        "id": fid,
        "filename": file.filename,
        "score": round(total, 3),
        "verdict": verdict,
        "breakdown": breakdown,
        "evidence": evidence_map,
        "transcript": redacted_text,
        "segments": redacted_segments,
        "transcript_chars": len(redacted_text),
        "model_info": {
            "whisper_model": os.environ.get("WHISPER_MODEL","base"),
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "use_llm_extractor": os.environ.get("USE_LLM_EXTRACTOR","false")
        }
    }
    json_path = os.path.join(OUTPUT_DIR, f"{fid}.json")
    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Append to CSV index
    with open(INDEX_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([fid, file.filename, round(total,3), verdict, len(redacted_text)])

    return JSONResponse(result)
