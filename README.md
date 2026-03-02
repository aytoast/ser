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

Built for the **Mistral AI Online Hackathon 2026** (W&B Fine-Tuning Track).

Ethos Studio is a full-stack emotional speech recognition platform combining real-time transcription, facial emotion recognition, and expressive audio tagging. It turns raw speech into richly annotated transcripts with emotions, non-verbal sounds, and delivery cues.

## Key Components

### Evoxtral — Expressive Tagged Transcription

LoRA finetune of [Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) that produces transcriptions with inline [ElevenLabs v3](https://elevenlabs.io/docs/api-reference/text-to-speech) audio tags. Two-stage pipeline: **SFT** (3 epochs) → **RL via RAFT** (rejection sampling, 1 epoch).

**Standard ASR:** `So I was thinking maybe we could try that new restaurant downtown.`

**Evoxtral:** `[nervous] So... [stammers] I was thinking maybe we could... [clears throat] try that new restaurant downtown? [laughs nervously]`

**Two model variants:**
- **[Evoxtral SFT](https://huggingface.co/YongkangZOU/evoxtral-lora)** — Best transcription accuracy (lowest WER)
- **[Evoxtral RL](https://huggingface.co/YongkangZOU/evoxtral-rl)** — Best expressive tag accuracy (highest Tag F1)

| Metric | Base Voxtral | Evoxtral SFT | Evoxtral RL | Best |
|--------|-------------|-------------|------------|------|
| **WER** ↓ | 6.64% | **4.47%** | 5.12% | SFT |
| **CER** ↓ | 2.72% | **1.23%** | 1.48% | SFT |
| **Tag F1** ↑ | 22.0% | 67.2% | **69.4%** | RL |
| **Tag Recall** ↑ | 22.0% | 69.4% | **72.7%** | RL |
| **Emphasis F1** ↑ | 42.0% | 84.0% | **86.0%** | RL |

- [SFT Model](https://huggingface.co/YongkangZOU/evoxtral-lora) | [RL Model](https://huggingface.co/YongkangZOU/evoxtral-rl)
- [Live Demo (HF Space)](https://huggingface.co/spaces/YongkangZOU/evoxtral)
- [API (Swagger UI)](https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run/docs)
- [W&B Dashboard](https://wandb.ai/yongkang-zou-ai/evoxtral)
- [Technical Report (PDF)](Evoxtral%20Technical%20Report.pdf) | [LaTeX source](docs/technical_report.tex)

### FER — Facial Emotion Recognition

MobileViT-XXS model trained on 8 emotion classes, exported to ONNX for real-time browser inference.

**Classes:** Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise

### Voxtral Server — Speech-to-Text + Emotion

Speech-to-text service with VAD sentence segmentation and per-segment emotion analysis, powered by [Voxtral Mini 4B](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).

## Architecture

```
Browser (port 3030)  →  Server layer (Node, :3000)  →  Model layer (Python, :8000)
      ↑ Studio UI            POST /api/speech-to-text          POST /transcribe
      ↑ Upload dialog        POST /api/transcribe-diarize      POST /transcribe-diarize
                             GET  /health                       GET  /health
```

| Layer | Path | Role |
|-------|------|------|
| **Model** | `model/voxtral-server` | Voxtral inference, VAD segmentation, emotion analysis |
| **Server** | `demo/server` | API entrypoint; proxies to Model |
| **Frontend** | `demo` | Next.js UI (upload, Studio editor, waveform, timeline) |
| **Evoxtral** | `training/scripts/` | Training, eval, RL, serving for expressive transcription |
| **FER** | `models/` | Facial emotion recognition ONNX model |

See [demo/README.md](demo/README.md) for full API and usage; [model/voxtral-server/README.md](model/voxtral-server/README.md) for the Model API.

## Project Structure

```
├── api/                    # Python FastAPI — local Voxtral inference + FER
├── proxy/                  # Node.js/Express — API gateway for frontend
├── web/                    # Next.js — Studio editor UI
├── training/               # Fine-tuning code (SFT + RL), data prep, eval
│   └── scripts/            # Modal scripts: train, RL (RAFT), eval, serve
├── space/                  # HuggingFace Space (Gradio demo)
├── models/                 # FER ONNX model (MobileViT-XXS)
├── docs/                   # Technical report, design docs, research refs
├── data/                   # Training data scripts (audio files gitignored)
└── Dockerfile              # Single-container HF Spaces build
```

## How to Run

**Requirements**: Python 3.10+, Node.js 20+, ffmpeg; GPU recommended.

### Model layer (port 8000)

```bash
cd model/voxtral-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Server layer (port 3000)

```bash
cd demo/server && npm install && npm run dev
```

### Frontend (port 3030)

```bash
cd demo && npm install && npm run dev
```

Open [http://localhost:3030](http://localhost:3030).

### Evoxtral API (Modal)

```bash
modal deploy training/scripts/serve_modal.py
```

## Tech Stack

- **Models**: Voxtral-Mini-3B + LoRA, Voxtral-Mini-4B, MobileViT-XXS
- **Training**: PyTorch, PEFT, Weights & Biases
- **Inference**: Modal (serverless GPU), HuggingFace ZeroGPU, ONNX Runtime
- **Backend**: FastAPI, Node.js
- **Frontend**: Next.js, Gradio

## Links

- [W&B Project](https://wandb.ai/yongkang-zou-ai/evoxtral) | [W&B Eval Report](https://wandb.ai/yongkang-zou-ai/evoxtral/reports/Evoxtral-—-Evaluation-Results:-Base-vs-SFT-vs-RL--VmlldzoxNjA3MzI3Nw==)
- [Evoxtral SFT Model](https://huggingface.co/YongkangZOU/evoxtral-lora) | [Evoxtral RL Model](https://huggingface.co/YongkangZOU/evoxtral-rl)
- [Evoxtral Demo](https://huggingface.co/spaces/YongkangZOU/evoxtral)
- [Evoxtral API (Swagger)](https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run/docs)
- [Technical Report (PDF)](Evoxtral%20Technical%20Report.pdf) | [LaTeX](docs/technical_report.tex)
