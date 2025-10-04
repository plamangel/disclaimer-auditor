# Disclaimer Auditor — AI-assisted compliance for call recordings

A small, local-first tool that transcribes a call recording, checks whether a specific disclaimer was delivered (even if paraphrased and in a different order), and produces a PASS/FAIL/NEEDS_REVIEW verdict with evidence.

**Highlights**
- Whisper-based speech-to-text (local) with timestamps.
- Hybrid compliance engine: keywords + multilingual sentence embeddings; optional LLM extractor.
- Policy-as-code in YAML (easy to edit).
- REST API (FastAPI) and a simple web UI for uploading audio.
- CLI scripts for batch processing.
- Outputs JSON and CSV summaries.

**What’s new**
- Optimized transcription with faster-whisper for improved speed.
- Added caching mechanism for LLM extraction to reduce redundant calls.
- Enhanced performance with timestamped transcripts for precise evidence.
- Improved logging and timing output for better monitoring.

## 1) Prerequisites

**OS:** macOS / Linux / Windows  
**Python:** 3.10+ recommended  
**FFmpeg:** required by Whisper

Install FFmpeg:
- macOS (Homebrew): `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Windows (Chocolatey): `choco install ffmpeg`

> If you don’t use a package manager, download FFmpeg from the official site and add it to PATH.

Optionally, for the LLM extractor:
- **OpenAI**: set `OPENAI_API_KEY` in `.env` (paid).
- **Ollama**: install and run a local model (e.g., `ollama run llama3.1`) and set `OLLAMA_MODEL` in `.env`.

## 2) Setup

```bash
git clone <your-repo-or-copy> disclaimer-auditor
cd disclaimer-auditor

# Use Makefile for setup
make setup

# Activate virtual environment
# macOS/Linux
source .venv/bin/activate
# Windows (Powershell)
# .venv\Scripts\Activate.ps1

# Optional: create cache folder for LLM extractor
mkdir -p .cache/llm
```

> First model downloads (Whisper & Sentence-Transformers) will take time and need internet access.

## 3) Quick test without audio (text transcript)

You can test the scoring engine using a plain text transcript:

```bash
python scripts/process_transcript.py sample_data/sample_transcript_ok.txt
python scripts/process_transcript.py sample_data/sample_transcript_fail.txt
```

Outputs will be saved under `outputs/` as JSON + a CSV index.

## 4) Process audio files (local CLI)

Place your `.wav/.mp3` files anywhere and run:

```bash
python scripts/process_audio.py /path/to/call1.mp3 /path/to/call2.wav
```

This will:
- Transcribe with Whisper (model controlled by `WHISPER_MODEL` in `.env`).
- Run compliance scoring against `policy/policy_disclaimer.yml`.
- Write `outputs/<basename>.json` and append to `outputs/index.csv`.

## 5) Run the API + Web UI

```bash
uvicorn app.main:app --reload
# Visit http://127.0.0.1:8000 in your browser
```

**Endpoints**
- `GET /` — simple upload UI
- `POST /analyze` — multipart form upload (audio file)
- `GET /health` — health probe

The UI displays the score, verdict, and per-requirement details with top evidence snippets.

## 6) Configuration

Edit `policy/policy_disclaimer.yml` to tweak requirements/weights.  
Environment variables in `.env`:

- `WHISPER_MODEL` (default: `base`): `tiny|base|small|medium|large`
- `USE_LLM_EXTRACTOR` (`true|false`): if `true`, the app will try LLM-based extraction
- `OPENAI_API_KEY` (optional): used if present and `USE_LLM_EXTRACTOR=true`
- `OLLAMA_MODEL` (optional): used if present and `USE_LLM_EXTRACTOR=true` (prefers Ollama over OpenAI)
- `LLM_CACHE_DIR` (optional): directory path for caching LLM extraction results (default: `.cache/llm`)
- `LLM_MAX_TRANSCRIPT_CHARS` (optional): max transcript length for LLM extraction to avoid timeouts (default: 3000)

## 7) Project Structure

New files added to support caching, embeddings, and logging:

- `app/embeddings.py` — handles multilingual sentence embeddings.
- `app/cache.py` — manages caching for LLM extraction.
- `app/logger.py` — centralized logging and timing utilities.

## 8) Performance and Logging

The application now outputs detailed timing logs for key steps such as transcription, embedding calculation, and LLM extraction. This helps monitor performance and identify bottlenecks during processing.

Example log output:

```
[INFO] Transcription completed in 12.3s
[INFO] Embeddings computed in 3.1s
[INFO] LLM extraction cached, skipped API call
```

## 9) Notes on diarization (optional)

This MVP analyzes the entire transcript (agent + client). In many pipelines, that’s sufficient because the disclaimer is usually spoken explicitly by the agent and contains distinctive phrases. You can later add speaker diarization (e.g., pyannote) to isolate the agent channel and feed only those segments to the scoring engine.

## 10) Git Workflow

We use a `develop` branch for ongoing development and feature branches for new work. Please create feature branches off `develop` and submit pull requests back to it for review.

## 11) Docker (optional)

A minimal Dockerfile is included. It installs Python deps and FFmpeg.

Build & run:

```bash
docker build -t disclaimer-auditor .
docker run --rm -it -p 8000:8000 -v "$(pwd)/outputs":/app/outputs -v "$(pwd)/.cache/llm":/app/.cache/llm --env-file .env disclaimer-auditor
# open http://127.0.0.1:8000
```

---

**License:** MIT
