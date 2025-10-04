import os
import csv
import uuid
import asyncio
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from dotenv import load_dotenv
from .logger import timed
from .embeddings import get_embed_model

from .policy_loader import load_policy
from .transcribe import transcribe_audio
from .llm_extractor import extract_flags
from .scoring import aggregate_score
from .utils import redact_pii, split_sentences_with_times, file_sha256

load_dotenv()

app = FastAPI(title="Disclaimer Auditor")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

POLICY_PATH = os.environ.get("POLICY_PATH", "policy/policy_disclaimer.yml")
policy = load_policy(POLICY_PATH)

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
async def analyze(
    request: Request,
    audio: UploadFile | None = File(None, description="Audio file (multipart/form-data)"),
    language: str | None = Form(None),
):
    # Defensive fallback to avoid 422-style issues when the form is empty or field name differs
    if audio is None:
        try:
            form = await request.form()
            for key in ("audio", "file", "upload", "data"):
                if key in form:
                    maybe = form.get(key)
                    if hasattr(maybe, "filename"):
                        audio = maybe  # type: ignore
                        break
        except Exception:
            pass

    if audio is None or not getattr(audio, "filename", ""):
        return JSONResponse({"error": "Missing 'audio' file in multipart/form-data."}, status_code=400)
    # Save temp file
    fid = str(uuid.uuid4())
    temp_path = os.path.join(OUTPUT_DIR, f"{fid}_{audio.filename}")
    with open(temp_path, "wb") as f:
        f.write(await audio.read())

    # Compute a stable digest and try full-pipeline cache
    digest = file_sha256(temp_path)
    cache_out = os.path.join(OUTPUT_DIR, f"{digest}.json")
    if os.path.exists(cache_out):
        with open(cache_out, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return JSONResponse(cached)

    # Transcribe (returns dict with transcript + segments)
    with timed("STT"):
        loop = asyncio.get_running_loop()
        tr = await loop.run_in_executor(None, lambda: transcribe_audio(temp_path, language=language))
    transcript_text = tr.get("transcript", "")
    segments = tr.get("segments", [])  # list of {text, start, end} (seconds)

    with timed("Redaction + sentence_times"):
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

    # Pre-encode all sentences once for faster similarity scoring
    # (used by scoring.py via the _pre_encoded bridge)
    embed_model = get_embed_model()
    with timed("Encode sentences"):
        sent_emb = embed_model.encode(sentences, normalize_embeddings=True)

    # Inject bundle into scoring config (without mutating the loaded policy)
    scoring_cfg = dict(policy.get("scoring", {}))
    scoring_cfg["_sentence_times_bundle"] = {"sentences": sentences, "times": times}
    scoring_cfg["_pre_encoded"] = {"sentences": sentences, "sent_emb": sent_emb}
    policy_for_scoring = {"requirements": policy["requirements"], "scoring": scoring_cfg}

    # Extract flags via LLM (optional) using redacted text only
    with timed("LLM extractor"):
        flags = extract_flags(redacted_text, policy)

    # Score using sentence-time bundle
    with timed("Scoring"):
        total, verdict, breakdown, evidence_map = aggregate_score(redacted_text, flags, policy_for_scoring, embed_model)

    # Persist JSON result
    result = {
        "id": fid,
        "filename": audio.filename,
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
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # Persist digest-based cache for identical files
    with open(cache_out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Append to CSV index
    with open(INDEX_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([fid, audio.filename, round(total,3), verdict, len(redacted_text)])

    return JSONResponse(result)
