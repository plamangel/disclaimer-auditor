# Disclaimer Auditor (MVP)

A small, local-first tool that transcribes a call recording, checks whether a specific disclaimer was delivered (even if paraphrased and in a different order), and produces a PASS/FAIL/NEEDS_REVIEW verdict with evidence.

**Highlights**
- Whisper-based speech-to-text (local) with timestamps.
- Hybrid compliance engine: keywords + multilingual sentence embeddings; optional LLM extractor.
- Policy-as-code in YAML (easy to edit).
- REST API (FastAPI) and a simple web UI for uploading audio.
- CLI scripts for batch processing.
- Outputs JSON and CSV summaries.

> All code, comments, and docs are intentionally in English.


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

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (Powershell)
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# edit .env to set USE_LLM_EXTRACTOR=true and add OPENAI_API_KEY or OLLAMA_MODEL (optional)
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


## 7) Notes on diarization (optional)

This MVP analyzes the entire transcript (agent + client). In many pipelines, that’s sufficient because the disclaimer is usually spoken explicitly by the agent and contains distinctive phrases. You can later add speaker diarization (e.g., pyannote) to isolate the agent channel and feed only those segments to the scoring engine.


## 8) Docker (optional)

A minimal Dockerfile is included. It installs Python deps and FFmpeg.
Build & run:

```bash
docker build -t disclaimer-auditor .
docker run --rm -it -p 8000:8000 -v "$(pwd)/outputs":/app/outputs --env-file .env disclaimer-auditor
# open http://127.0.0.1:8000
```


## 9) Troubleshooting

- **Whisper complains about FFmpeg**: ensure FFmpeg is installed and on PATH.
- **Torch install issues**: try upgrading pip; on Apple Silicon use a recent Python and pip.
- **Models not downloading**: verify internet access; corporate proxies may need configuration.
- **Slow first run**: model downloads and caching happen one-time.

---

**License:** MIT
