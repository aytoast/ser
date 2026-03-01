# Server Layer

Node proxy service. Exposes the client-facing API and forwards requests to the **Model layer** (voxtral-server).

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

Simple transcription. Forwarded to Model layer `POST /transcribe`. Timeout: **5 min**.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (wav, mp3, flac, ogg, m4a, webm) |
| **Limits** | ≤ 100 MB |

**Response (200)**

```json
{
  "text": "transcribed text",
  "words": [],
  "languageCode": null
}
```

**Errors**

| Status | Body |
|--------|------|
| 400 | `{"error": "Upload an audio file (form field: audio)"}` |
| 502 | Model layer error or unreachable |
| 504 | `{"error": "Request timeout (>5 min); try shorter audio"}` |

---

### POST /api/transcribe-diarize

Transcription + speaker diarization + emotion analysis. Forwarded to Model layer `POST /transcribe-diarize`. Timeout: **10 min**.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (wav, mp3, flac, ogg, m4a, webm) |
| **Query** | `num_speakers` (optional, integer 1–10) — speaker count hint; 0 = auto |
| **Limits** | ≤ 100 MB |

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
  "filename": "recording.m4a",
  "diarization_method": "vad_mfcc"
}
```

**Errors**

| Status | Body |
|--------|------|
| 400 | `{"error": "Upload an audio file (form field: audio)"}` |
| 502 | Model layer error or unreachable |
| 504 | `{"error": "Request timeout (>10 min); try shorter audio"}` |

---

### GET /health

Proxies `GET {MODEL_URL}/health` and wraps it.

**Response (200)**

```json
{
  "ok": true,
  "server": "ser-server",
  "model": {
    "status": "ok",
    "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
    "model_loaded": true,
    "ffmpeg": true,
    "pyannote_available": false,
    "hf_token_set": false,
    "max_upload_mb": 100
  }
}
```

**Response (502)** — when Model layer is unreachable:

```json
{"ok": false, "error": "Cannot reach Model layer; start model/voxtral-server first", "url": "http://127.0.0.1:8000"}
```

---

## Usage examples

```bash
# Health
curl -s http://localhost:3000/health

# Transcribe
curl -X POST http://localhost:3000/api/speech-to-text -F "audio=@./recording.m4a"

# Transcribe + diarize + emotion
curl -X POST http://localhost:3000/api/transcribe-diarize -F "audio=@./recording.m4a"

# With speaker count hint
curl -X POST "http://localhost:3000/api/transcribe-diarize?num_speakers=2" -F "audio=@./recording.m4a"
```
