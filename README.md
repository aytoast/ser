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

## Studio features

### Transcript editor
- **Character-level text highlighting** — transcript text sweeps dark→gray in sync with playback position, character by character
- **Click-to-seek** — click any character in the transcript to jump the timeline to that exact moment; uses `caretRangeFromPoint` for precision
- **Inline `[bracket]` badges** — paralinguistic tags produced by Voxtral (e.g. `[laughs]`, `[sighs]`) render as pill badges at their exact inline position, not appended at the end; clicking a badge seeks to the moment just before it
- **Bidirectional timeline ↔ transcript sync** — scrolling/clicking the timeline highlights the active segment in the transcript and auto-scrolls it into view; clicking a segment row seeks the timeline
- **Per-segment state** (`past` / `active` / `future`) with opacity transitions

### Live emotion panel (right sidebar)
- **Streaming speech emotion** — the Speech emotion badge updates sub-segment as playback passes each `[bracket]` tag; timing is estimated from the tag's character position proportional to segment duration
- **Streaming valence & arousal bars** — both bars transition to the bracket tag's valence/arousal values at the same moment, creating a continuous emotional arc within each segment
- **Per-second face emotion** (video only) — the Face badge updates every second from the `face_emotion_timeline` returned by the FER pipeline, more granular than the per-segment majority vote
- **Live indicator** — animated green dot appears during playback

### Timeline
- **Click-to-seek** on the track area
- **Active segment highlight** with ring indicator
- **Played-region overlay** — subtle tint left of the playhead
- **Dot + line playhead** design

### Video support
- Video files (`.mp4`, `.mkv`, `.avi`, `.mov`, `.m4v`, `.webm`) display inline in the right panel preview area
- FER runs on video frames and produces both per-segment majority-vote emotion and a per-second `face_emotion_timeline`

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
    ↕  Next.js UI (upload, Studio editor, timeline, live emotion panel)
Node proxy (:3000)
    ↕  Express — streams multipart upload, manages session state
Python API (:8000)
    ├─ POST /transcribe-diarize  — VAD + Voxtral STT + emotion tags + FER timeline
    └─ POST /fer                 — per-frame FER via MobileViT-XXS ONNX
```

nginx on port 7860 (HF Spaces public port) routes:
- `/api/*` → Node proxy `:3000`
- `/_next/*`, `/` → Next.js `:3030`

| Directory | Port | Role |
|-----------|------|------|
| `api/`    | 8000 | Voxtral local inference; VAD segmentation; per-segment emotion; FER timeline |
| `proxy/`  | 3000 | API entrypoint; proxies to `api/` |
| `web/`    | 3030 | Next.js Studio UI |

## API response format

`POST /api/transcribe-diarize` returns:

```json
{
  "filename": "interview.mp4",
  "duration": 42.5,
  "text": "Full transcript...",
  "segments": [
    {
      "id": 1,
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 5.2,
      "text": "Welcome to the show. [laughs]",
      "emotion": "Happy",
      "valence": 0.7,
      "arousal": 0.6,
      "face_emotion": "Happy"
    }
  ],
  "has_video": true,
  "face_emotion_timeline": {
    "0": "Neutral",
    "1": "Happy",
    "2": "Happy"
  }
}
```

`face_emotion_timeline` maps each second (as a string key) to the majority FER label for that second. Only present for video inputs.

## Bracket tag emotions

Voxtral produces paralinguistic `[bracket]` tags in transcriptions. The frontend and API both recognise these tags and map them to `(emotion, valence, arousal)` triples:

| Tag | Emotion | Valence | Arousal |
|-----|---------|---------|---------|
| `[laughs]` / `[laughing]` | Happy | +0.70 | +0.60 |
| `[sighs]` / `[sighing]` | Sad | −0.30 | −0.30 |
| `[whispers]` / `[whispering]` | Calm | +0.10 | −0.50 |
| `[shouts]` / `[shouting]` | Angry | −0.50 | +0.80 |
| `[exclaims]` | Excited | +0.50 | +0.70 |
| `[gasps]` | Surprised | +0.20 | +0.70 |
| `[hesitates]` / `[stutters]` / `[stammers]` | Anxious | −0.20 | +0.35 |
| `[cries]` / `[crying]` | Sad | −0.70 | +0.40 |
| `[claps]` / `[applause]` | Happy | +0.60 | +0.50 |
| `[clears throat]` / `[pause]` | Neutral | 0.00 | ±0.10 |

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

Inference is optimised with `merge_and_unload()` (removes PEFT per-forward overhead), `torch.set_num_threads(cpu_count)`, and `torch.inference_mode()`.

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

Upload a video to also get per-segment facial emotion and the `face_emotion_timeline`.

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
>
> ```bash
> MODELS_SHA=$(git ls-tree space/main | grep $'\tmodels$' | awk '{print $3}')
> TREE_SHA=$((git ls-tree HEAD | grep -v $'\tmodels$'; echo "040000 tree $MODELS_SHA\tmodels") | git mktree)
> PARENT=$(git rev-parse space/main)
> COMMIT_SHA=$(git commit-tree "$TREE_SHA" -p "$PARENT" -m "your message")
> git push space "${COMMIT_SHA}:refs/heads/main"
> ```
