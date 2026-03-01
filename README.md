---
title: Ethos Studio
emoji: 🎤
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Ethos Studio — Emotional Speech Recognition

Speech-to-text with VAD sentence segmentation, per-segment emotion tagging, and facial emotion recognition (FER), powered by a fine-tuned [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) + [evoxtral-lora](https://huggingface.co/YongkangZOU/evoxtral-lora) running locally, and a MobileViT-XXS ONNX model for FER.

## Repository layout

```
ser/
├── api/              # Python FastAPI — local Voxtral inference + FER pipeline
├── proxy/            # Node.js/Express — API gateway for the frontend
├── web/              # Next.js — Studio editor UI
├── training/         # Fine-tuning code (Voxtral LoRA), data prep, eval
├── docs/             # Specs, model card, hackathon guidelines
├── models/           # ONNX weights (emotion_model_web.onnx — tracked via Git LFS)
├── Dockerfile        # Single-container HF Spaces build
├── nginx.conf        # Reverse proxy config (port 7860 → :3000/:3030)
└── supervisord.conf  # Process manager for all four services
```

## Architecture

```
Browser (:3030)
    ↕  Next.js UI (upload, Studio editor, waveform, timeline, FER badges)
Node proxy (:3000)
    ↕  Express — streams multipart upload, manages session state
Python API (:8000)
    ├─ POST /transcribe-diarize  — VAD + Voxtral STT + emotion tags
    └─ POST /fer                 — per-frame FER via MobileViT-XXS ONNX
```

nginx on port 7860 (HF Spaces public port) routes:
- `/api/*` → Node proxy `:3000`
- `/_next/*`, `/` → Next.js `:3030`

| Directory | Port | Role |
|-----------|------|------|
| `api/`    | 8000 | Voxtral local inference; VAD segmentation; per-segment emotion; FER |
| `proxy/`  | 3000 | API entrypoint; proxies to `api/` |
| `web/`    | 3030 | Next.js Studio UI |

## How to run locally

**Requirements**: Python 3.11+, Node.js 22+, ffmpeg, a GPU with ~8 GB VRAM (or CPU fallback).

### 1. Python API (port 8000)

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On first start the Voxtral model (~6 GB) is downloaded from HuggingFace. Set `MODEL_ID` / `ADAPTER_ID` env vars to override.

### 2. Node proxy (port 3000)

```bash
cd proxy
npm install
npm run dev
```

### 3. Next.js frontend (port 3030)

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3030](http://localhost:3030).

### Quick health check

```bash
curl -s http://localhost:3000/health
curl -X POST http://localhost:3000/api/transcribe-diarize -F "audio=@/path/to/audio.m4a"
curl -X POST http://localhost:3000/api/transcribe-diarize -F "audio=@/path/to/video.mov"
```

Upload a video to also get per-segment facial emotion (`face_emotion` field).

## Models

| Model | Purpose | Source |
|-------|---------|--------|
| `mistralai/Voxtral-Mini-3B-2507` | Speech-to-text base | HF Hub (downloaded at runtime) |
| `YongkangZOU/evoxtral-lora` | LoRA adapter — emotion-aware transcription | HF Hub (downloaded at runtime) |
| `models/emotion_model_web.onnx` | MobileViT-XXS 8-class FER | Stored in repo via Git LFS |

FER emotion classes: `Anger | Contempt | Disgust | Fear | Happy | Neutral | Sad | Surprise`

## Training

Fine-tuning scripts and data utilities live in `training/`:

```bash
cd training
pip install -r requirements.txt
# fine-tune on Modal
python scripts/train_modal.py
# push adapter to HF Hub
python scripts/push_hub.py
```

See `training/finetune.py` for the full training loop and `training/config.py` for hyperparameters.

## Deployment (HuggingFace Spaces)

The app runs as a single Docker container. `supervisord` starts all four processes; nginx handles port routing.

```bash
# Build and test locally
docker build -t ethos-studio .
docker run -p 7860:7860 ethos-studio
```

HF Spaces auto-builds on every push to the `main` branch of the linked Space.

> **Note**: `models/emotion_model_web.onnx` is stored via Git LFS / Xet on HF Spaces. When pushing a new commit, use the `commit-tree` graft technique to reuse the existing LFS tree rather than re-uploading the binary.
