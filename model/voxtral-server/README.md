# Voxtral Speech-to-Text (Model Layer)

Local inference API based on [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602). Provides transcription, VAD sentence segmentation, and per-segment emotion analysis.

**Requirements**: Python 3.10+; GPU ≥16GB VRAM recommended; system **ffmpeg** (e.g. `brew install ffmpeg`).

---

## Startup

```bash
cd model/voxtral-server
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Default port: **8000**. First run may download the model (~8–16GB). Wait for `Application startup complete`.

---

## API

### POST /transcribe

Simple transcription (no diarization).

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (required) |
| **Formats** | wav, mp3, flac, ogg, m4a, webm |
| **Max size** | `MAX_UPLOAD_MB` (default 100 MB) |

**Response (200)**

```json
{
  "text": "transcribed text",
  "words": [],
  "languageCode": null
}
```

**Errors**

| Status | Meaning |
|--------|---------|
| 400 | No/invalid file, empty, or unsupported format |
| 413 | File exceeds `MAX_UPLOAD_MB` |

---

### POST /transcribe-diarize

Full pipeline: transcription + VAD sentence segmentation + per-segment emotion analysis.
All segments are labelled `SPEAKER_00` (single-speaker mode).

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (required) |
| **Formats** | wav, mp3, flac, ogg, m4a, webm |
| **Max size** | `MAX_UPLOAD_MB` (default 100 MB) |

Segmentation: silence gaps ≥ 0.3 s create a new segment; gaps < 0.3 s are merged (intra-phrase pauses). Text is distributed to segments by sentence boundaries (punctuation marks), with character-level fallback.

**Response (200)**

```json
{
  "segments": [
    {
      "id": 1,
      "speaker": "SPEAKER_00",
      "start": 0.96,
      "end": 3.23,
      "text": "你好嗎?我要是出軌了你會怎麼辦?",
      "emotion": "Frustrated",
      "valence": -0.33,
      "arousal": 0.28
    }
  ],
  "duration": 5.65,
  "text": "full transcript",
  "filename": "audio.m4a",
  "diarization_method": "vad"
}
```

`diarization_method`: `"vad"`.

**Errors**

| Status | Meaning |
|--------|---------|
| 400 | No/invalid file, empty, or unsupported format |
| 413 | File exceeds `MAX_UPLOAD_MB` |

---

### GET /health

**Response (200)**

```json
{
  "status": "ok",
  "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
  "model_loaded": true,
  "ffmpeg": true,
  "pyannote_available": false,
  "hf_token_set": false,
  "max_upload_mb": 100
}
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXTRAL_MODEL_ID` | `mistralai/Voxtral-Mini-4B-Realtime-2602` | Hugging Face model ID |
| `MAX_UPLOAD_MB` | `100` | Max upload size in MB |

---

## Usage examples

```bash
# Simple transcription
curl -X POST http://127.0.0.1:8000/transcribe -F "audio=@audio.m4a"

# Transcription + VAD segmentation + emotion analysis
curl -X POST http://127.0.0.1:8000/transcribe-diarize -F "audio=@audio.m4a"

# Health
curl -s http://127.0.0.1:8000/health
```
