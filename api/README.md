# API Layer (Python FastAPI — port 8000)

Local Voxtral inference pipeline. Loads `mistralai/Voxtral-Mini-3B-2507` + `YongkangZOU/evoxtral-lora` (PEFT adapter) locally, runs VAD sentence segmentation, per-segment emotion tagging, and facial emotion recognition (FER) for video inputs.

**Requirements**: Python 3.11+, system **ffmpeg** (`brew install ffmpeg` / `apt install ffmpeg`). GPU with ~8 GB VRAM recommended; CPU fallback supported (expect ~50 s per audio second).

---

## Startup

```bash
cd api
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Default port: **8000**. On first start the Voxtral model (~8 GB total) is downloaded from HuggingFace. Set `HF_HUB_DISABLE_XET=1` if download stalls behind a local proxy.

---

## API

### POST /transcribe

Simple transcription. Audio is converted to WAV and passed to the local Voxtral model.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio or video file (required) |
| **Formats** | wav, mp3, flac, ogg, m4a, webm, mp4, mov, mkv |
| **Max size** | `MAX_UPLOAD_MB` (default 100 MB) |

**Response (200)**

```json
{
  "text": "Hello! [laughs] How are you?",
  "words": [],
  "languageCode": "en"
}
```

---

### POST /transcribe-diarize

Full pipeline: local Voxtral STT + VAD sentence segmentation + per-segment emotion tagging. For video inputs, also runs per-frame FER via MobileViT-XXS ONNX and adds `face_emotion` per segment.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio or video file (required) |
| **Formats** | wav, mp3, flac, ogg, m4a, webm, mp4, mov, mkv |
| **Max size** | `MAX_UPLOAD_MB` (default 100 MB) |

Segmentation: silence gaps ≥ 0.3 s create a new segment; gaps < 0.3 s are merged.

**Response (200) — audio input**

```json
{
  "segments": [
    {
      "id": 1,
      "speaker": "SPEAKER_00",
      "start": 0.96,
      "end": 3.23,
      "text": "Hello! [laughs] How are you?",
      "emotion": "Happy",
      "valence": 0.7,
      "arousal": 0.6
    }
  ],
  "duration": 5.65,
  "text": "Hello! [laughs] How are you?",
  "filename": "audio.m4a",
  "diarization_method": "vad",
  "has_video": false
}
```

**Response (200) — video input** (adds `face_emotion` per segment)

```json
{
  "segments": [
    {
      "id": 1,
      "speaker": "SPEAKER_00",
      "start": 0.96,
      "end": 3.23,
      "text": "Hello!",
      "emotion": "Happy",
      "valence": 0.7,
      "arousal": 0.6,
      "face_emotion": "Happy"
    }
  ],
  "duration": 5.65,
  "text": "Hello!",
  "filename": "video.mov",
  "diarization_method": "vad",
  "has_video": true
}
```

`face_emotion` values: `Anger | Contempt | Disgust | Fear | Happy | Neutral | Sad | Surprise`

**Errors**

| Status | Meaning |
|--------|---------|
| 400 | No/invalid file, empty, or unsupported format |
| 413 | File exceeds `MAX_UPLOAD_MB` |
| 500 | Transcription or inference error |

---

### GET /health

**Response (200)**

```json
{
  "status": "ok",
  "model": "mistralai/Voxtral-Mini-3B-2507 + YongkangZOU/evoxtral-lora (local)",
  "model_loaded": true,
  "ffmpeg": true,
  "fer_enabled": true,
  "device": "cpu",
  "max_upload_mb": 100
}
```

---

### GET /debug-inference

Smoke-test endpoint: synthesizes 0.5 s of silence and runs a minimal `generate()` call. Useful for verifying the model is loaded and functional without uploading a real file.

**Response (200)**

```json
{
  "ok": true,
  "text": "",
  "dtype": "torch.bfloat16",
  "device": "cpu"
}
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `mistralai/Voxtral-Mini-3B-2507` | Base Voxtral model on HF Hub |
| `ADAPTER_ID` | `YongkangZOU/evoxtral-lora` | PEFT LoRA adapter on HF Hub |
| `FER_MODEL_PATH` | (auto-detected) | Path to `emotion_model_web.onnx`; auto-detects `/app/models/` (Docker) and `../models/` (local) |
| `MAX_UPLOAD_MB` | `100` | Max upload size in MB |

---

## Usage examples

```bash
# Health
curl -s http://127.0.0.1:8000/health

# Smoke-test inference
curl -s http://127.0.0.1:8000/debug-inference

# Simple transcription
curl -X POST http://127.0.0.1:8000/transcribe -F "audio=@audio.m4a"

# Full pipeline (audio)
curl -X POST http://127.0.0.1:8000/transcribe-diarize -F "audio=@audio.m4a"

# Full pipeline (video — also returns face_emotion per segment)
curl -X POST http://127.0.0.1:8000/transcribe-diarize -F "audio=@video.mov"
```
