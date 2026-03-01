# Proxy Layer (Node/Express — port 3000)

API gateway. Accepts multipart file uploads from the browser, forwards them to the **API layer** (Python FastAPI on port 8000), and returns JSON responses.

- **Port**: `3000` (override with `PORT`)
- **API layer URL**: `http://127.0.0.1:8000` (override with `MODEL_URL`)

---

## Startup

```bash
cd proxy
npm install
npm run dev    # dev with --watch
# or
npm start
```

Requires **Node.js 22+**.

---

## API

### POST /api/speech-to-text

Simple transcription. Forwarded to API layer `POST /transcribe`. Timeout: **30 min** (CPU inference is slow).

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
  "languageCode": "en"
}
```

**Errors**

| Status | Body |
|--------|------|
| 400 | `{"error": "Upload an audio file (form field: audio)"}` |
| 502 | API layer error or unreachable |
| 504 | `{"error": "Request timeout (>30 min); try shorter audio"}` |

---

### POST /api/transcribe-diarize

Full pipeline: transcription + VAD sentence segmentation + emotion analysis. For video inputs, also returns `face_emotion` per segment. Forwarded to API layer `POST /transcribe-diarize`. Timeout: **60 min**.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio or video file (wav, mp3, flac, ogg, m4a, webm, mp4, mov, mkv) |
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
      "emotion": "Happy",
      "valence": 0.7,
      "arousal": 0.6,
      "face_emotion": "Happy"
    }
  ],
  "duration": 42.3,
  "text": "full transcript",
  "filename": "recording.mov",
  "diarization_method": "vad",
  "has_video": true
}
```

`face_emotion` is present only when a video file is uploaded and FER is enabled. `has_video` indicates whether facial emotion recognition ran.

**Errors**

| Status | Body |
|--------|------|
| 400 | `{"error": "Upload an audio file (form field: audio)"}` |
| 502 | API layer error or unreachable |
| 504 | `{"error": "Request timeout (>60 min); try shorter audio"}` |

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
    "model": "mistralai/Voxtral-Mini-3B-2507 + YongkangZOU/evoxtral-lora (local)",
    "model_loaded": true,
    "ffmpeg": true,
    "fer_enabled": true,
    "device": "cpu",
    "max_upload_mb": 100
  }
}
```

**Response (502)** — when API layer is unreachable:

```json
{"ok": false, "error": "Cannot reach Model layer; start model/voxtral-server first", "url": "http://127.0.0.1:8000"}
```

---

### GET /api/debug-inference

Proxies `GET {MODEL_URL}/debug-inference` — smoke-tests the local Voxtral model with a short silence clip.

---

## Usage examples

```bash
# Health
curl -s http://localhost:3000/health

# Transcribe (audio)
curl -X POST http://localhost:3000/api/speech-to-text -F "audio=@./recording.m4a"

# Transcribe + segment + emotion (audio or video)
curl -X POST http://localhost:3000/api/transcribe-diarize -F "audio=@./recording.m4a"
curl -X POST http://localhost:3000/api/transcribe-diarize -F "audio=@./video.mov"
```
