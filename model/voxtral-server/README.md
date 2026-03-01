# Voxtral Speech-to-Text (Model Layer)

Local inference API based on [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602). Used by the **Server layer** or callable directly. Offline transcription: upload audio, get full text in one response.

**Requirements**: **Python 3.10+**; recommended GPU ≥16GB VRAM; system **ffmpeg** (e.g. `brew install ffmpeg`).

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

Transcribe an audio file.

| | |
|--|--|
| **Method** | `POST` |
| **Path** | `/transcribe` |
| **Content-Type** | `multipart/form-data` |
| **Request body** | Single field **`audio`**: the audio file (required) |
| **Supported formats** | wav, mp3, flac, ogg, m4a, webm |
| **Max size** | Default 100 MB (set by `MAX_UPLOAD_MB` env) |

**Response (200)**  
JSON:

```json
{
  "text": "transcribed text",
  "words": [],
  "languageCode": null
}
```

**Error responses**

| Status | Meaning |
|--------|--------|
| 400 | No/invalid file, empty file, or unsupported/corrupt audio |
| 413 | File exceeds `MAX_UPLOAD_MB` |

---

### GET /health

Health and dependency check.

**Response (200)**  
JSON:

```json
{
  "status": "ok",
  "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
  "model_loaded": true,
  "ffmpeg": "/opt/homebrew/bin/ffmpeg",
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

## Usage example

```bash
# Transcribe (direct to Model layer)
curl -X POST http://127.0.0.1:8000/transcribe -F "audio=@/path/to/audio.m4a"

# Health
curl -s http://127.0.0.1:8000/health
```

Example success response:

```json
{"text":"Hello world","words":[],"languageCode":null}
```
