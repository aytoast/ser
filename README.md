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

LoRA finetune of [Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) that produces transcriptions with inline [ElevenLabs v3](https://elevenlabs.io/docs/api-reference/text-to-speech) audio tags.

**Standard ASR:** `So I was thinking maybe we could try that new restaurant downtown.`

**Evoxtral:** `[nervous] So... I was thinking maybe we could [clears throat] try that new restaurant downtown? [laughs nervously]`

| Metric | Base Voxtral | Evoxtral | Improvement |
|--------|-------------|----------|-------------|
| WER | 6.64% | **4.47%** | 32.7% better |
| Tag F1 | 22.0% | **67.2%** | 3x better |

- [Model on HuggingFace](https://huggingface.co/YongkangZOU/evoxtral-lora)
- [Live Demo (HF Space)](https://huggingface.co/spaces/YongkangZOU/evoxtral)
- [API (Swagger UI)](https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run/docs)
- [W&B Dashboard](https://wandb.ai/yongkang-zou-ai/evoxtral)

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
| **Evoxtral** | `scripts/` | Training, eval, serving for expressive transcription |
| **FER** | `models/` | Facial emotion recognition ONNX model |

See [demo/README.md](demo/README.md) for full API and usage; [model/voxtral-server/README.md](model/voxtral-server/README.md) for the Model API.

## Project Structure

```
├── model/voxtral-server/   # Voxtral inference server (Python/FastAPI)
├── demo/                   # Next.js frontend + Node server
├── scripts/                # Evoxtral training, eval, serving (Modal)
├── src/                    # Data pipeline, tag taxonomy, eval metrics
├── space/                  # HuggingFace Space (Gradio demo)
├── models/                 # FER ONNX model
├── model_card/             # HuggingFace model card
├── docs/                   # Design docs and research
└── data/                   # Training data scripts (audio files gitignored)
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
modal deploy scripts/serve_modal.py
```

## Tech Stack

- **Models**: Voxtral-Mini-3B + LoRA, Voxtral-Mini-4B, MobileViT-XXS
- **Training**: PyTorch, PEFT, Weights & Biases
- **Inference**: Modal (serverless GPU), HuggingFace ZeroGPU, ONNX Runtime
- **Backend**: FastAPI, Node.js
- **Frontend**: Next.js, Gradio

## Links

- [W&B Project](https://wandb.ai/yongkang-zou-ai/evoxtral)
- [Evoxtral Model](https://huggingface.co/YongkangZOU/evoxtral-lora)
- [Evoxtral Demo](https://huggingface.co/spaces/YongkangZOU/evoxtral)
- [Evoxtral API](https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run/docs)
