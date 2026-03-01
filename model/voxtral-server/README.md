# Voxtral Speech-to-Text (Model Layer)

Local inference API based on [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602). Provides transcription, speaker diarization, and per-segment emotion analysis.

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

Optional: set `HF_TOKEN` for real speaker diarization via pyannote:

```bash
export HF_TOKEN=hf_...
```

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

Full pipeline: transcription + speaker diarization + emotion analysis per segment.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (required) |
| **Query** | `num_speakers` (int, 0–10, default 0) — speaker count hint; 0 = auto-detect |
| **Formats** | wav, mp3, flac, ogg, m4a, webm |
| **Max size** | `MAX_UPLOAD_MB` (default 100 MB) |

Speaker detection (priority order):
1. **pyannote/speaker-diarization-3.1** — requires `HF_TOKEN` env var + `pyannote.audio` installed
2. **VAD + MFCC KMeans** — silence-split fallback; always available

**Response (200)**

```json
{
  "segments": [
    {
      "id": 1,
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 4.2,
      "text": "Hello, how are you?",
      "emotion": "neutral",
      "valence": 0.1,
      "arousal": 0.2
    }
  ],
  "duration": 42.3,
  "text": "full transcript",
  "filename": "audio.m4a",
  "diarization_method": "vad_mfcc"
}
```

`diarization_method`: `"pyannote"` or `"vad_mfcc"`.

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
| `HF_TOKEN` | _(unset)_ | Hugging Face token; enables pyannote speaker diarization |

---

## Usage examples

```bash
# Simple transcription
curl -X POST http://127.0.0.1:8000/transcribe -F "audio=@audio.m4a"

# Transcription + diarization + emotion analysis
curl -X POST http://127.0.0.1:8000/transcribe-diarize -F "audio=@audio.m4a"

# With speaker count hint
curl -X POST "http://127.0.0.1:8000/transcribe-diarize?num_speakers=2" -F "audio=@audio.m4a"

# Health
curl -s http://127.0.0.1:8000/health
```
