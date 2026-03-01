# Server Layer

Node service that exposes a single API and proxies transcription requests to the **Model layer** (voxtral-server).

- **Port**: `3000` (override with `PORT`)
- **Model layer URL**: `http://127.0.0.1:8000` (override with `MODEL_URL`)

---

## Startup

```bash
npm install
npm run dev    # dev with --watch
# or
npm start
```

Requires **Node.js 18+**.

---

## API

### POST /api/speech-to-text

Transcribe an audio file. Request is forwarded to Model layer `POST /transcribe`.

| | |
|--|--|
| **Method** | `POST` |
| **Path** | `/api/speech-to-text` |
| **Content-Type** | `multipart/form-data` |
| **Request body** | Single field **`audio`**: the audio file (any of wav, mp3, flac, ogg, m4a, webm) |
| **Limits** | Max file size 100 MB; request timeout 5 minutes |

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

| Status | Body |
|--------|------|
| 400 | `{"error": "Upload an audio file (form field: audio)"}` or file too large |
| 502 | Model layer error or unreachable |
| 504 | `{"error": "Transcription timeout (over 5 min); try shorter audio or retry later"}` |

---

### GET /health

Health check; also returns Model layer status (proxied from `GET {MODEL_URL}/health`).

**Response (200)**  
e.g.:

```json
{
  "ok": true,
  "server": "ser-server",
  "model": {
    "status": "ok",
    "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
    "model_loaded": true,
    "ffmpeg": "/opt/homebrew/bin/ffmpeg",
    "max_upload_mb": 100
  }
}
```

**Response (502)**  
When Model layer is unreachable: `{"ok":false,"error":"...","url":"http://127.0.0.1:8000"}`.

---

## Usage example

```bash
# Transcribe
curl -X POST http://localhost:3000/api/speech-to-text -F "audio=@./recording.m4a"

# Health
curl -s http://localhost:3000/health
```
