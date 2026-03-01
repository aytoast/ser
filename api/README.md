# Evoxtral Speech-to-Text (Model Layer)

Proxy API that forwards audio to the external [YongkangZOU/evoxtral-lora](https://huggingface.co/YongkangZOU/evoxtral-lora) inference endpoint (hosted on Modal), then adds VAD sentence segmentation and emotion parsing from inline expression tags.

**Requirements**: Python 3.10+, system **ffmpeg** (`brew install ffmpeg` / `apt install ffmpeg`). No GPU or local model download needed.

---

## Startup

```bash
cd model/voxtral-server
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Default port: **8000**. Starts instantly — inference is handled remotely.

---

## API

### POST /transcribe

Simple transcription. Audio is converted to WAV and forwarded to the evoxtral API.
The transcription text may include inline expression tags like `[laughs]`, `[sighs]`, `[whispers]`.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (required) |
| **Formats** | wav, mp3, flac, ogg, m4a, webm |
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

Full pipeline: transcription (via evoxtral API) + VAD sentence segmentation + per-segment emotion parsing.
All segments are labelled `SPEAKER_00` (single-speaker mode).
Emotion is derived from evoxtral's inline expression tags (`[laughs]` → Happy, `[sighs]` → Sad, etc.).

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (required) |
| **Formats** | wav, mp3, flac, ogg, m4a, webm |
| **Max size** | `MAX_UPLOAD_MB` (default 100 MB) |

Segmentation: silence gaps ≥ 0.3 s create a new segment; gaps < 0.3 s are merged. Text is distributed by sentence boundaries, with character-level fallback.

**Response (200)**

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
  "diarization_method": "vad"
}
```

`diarization_method`: always `"vad"`.

**Errors**

| Status | Meaning |
|--------|---------|
| 400 | No/invalid file, empty, or unsupported format |
| 413 | File exceeds `MAX_UPLOAD_MB` |
| 502 | Evoxtral external API unreachable or returned an error |

---

### GET /health

**Response (200)**

```json
{
  "status": "ok",
  "model": "YongkangZOU/evoxtral-lora (external API)",
  "model_loaded": true,
  "ffmpeg": true,
  "pyannote_available": false,
  "hf_token_set": false,
  "max_upload_mb": 100,
  "evoxtral_api": "https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run"
}
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVOXTRAL_API` | `https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run` | External evoxtral inference endpoint |
| `MAX_UPLOAD_MB` | `100` | Max upload size in MB |

---

## Usage examples

```bash
# Simple transcription
curl -X POST http://127.0.0.1:8000/transcribe -F "audio=@audio.m4a"

# Transcription + VAD segmentation + emotion
curl -X POST http://127.0.0.1:8000/transcribe-diarize -F "audio=@audio.m4a"

# Health
curl -s http://127.0.0.1:8000/health
```
