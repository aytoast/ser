# demo — Frontend & Server Layer

This folder contains the Next.js frontend (Ethos Studio) and the Node proxy server.

## Architecture

```
Browser (port 3030)
  → Server layer (Node, port 3000)   POST /api/speech-to-text, POST /api/transcribe-diarize, GET /health
      → Model layer (Python, port 8000)   POST /transcribe, POST /transcribe-diarize, GET /health
```

- **Frontend** (`demo/`): Upload page + Studio editor. Calls the Server layer. See [Startup → 3](#3-frontend-nextjs-port-3030).
- **Server layer** (`demo/server/`): Proxies client requests to the Model layer. See [demo/server/README.md](server/README.md) for API details.
- **Model layer** (`model/voxtral-server/`): Voxtral inference + diarization + emotion analysis. See [model/voxtral-server/README.md](../model/voxtral-server/README.md) for API details.

---

## Startup

### 1. Model layer (Python, port 8000)

Requires **Python 3.10+** and **ffmpeg**.

```bash
cd ../model/voxtral-server
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

First run may download the model (~8–16GB). Optional: set `HF_TOKEN` for pyannote-based speaker diarization.

### 2. Server layer (Node, port 3000)

```bash
cd server
npm install
npm run dev
```

### 3. Frontend (Next.js, port 3030)

```bash
cd ..   # back to demo/
npm install
npm run dev
```

Open [http://localhost:3030](http://localhost:3030).

- **Home page**: Click **Transcribe files**, drag-drop an audio file, choose language and options, then **Upload**. The file is sent to `/api/transcribe-diarize` and results open in the Studio.
- **Studio page** (`/studio`): Three-column layout — transcript segments (speaker + emotion badges) on the left, waveform in the center, audio player on the right.

### 4. Quick check (API only)

```bash
curl -s http://localhost:3000/health
curl -X POST http://localhost:3000/api/speech-to-text -F "audio=@audio.m4a"
curl -X POST http://localhost:3000/api/transcribe-diarize -F "audio=@audio.m4a"
```

---

## API (Server layer)

Clients should call the **Server layer** only. The Model layer is used internally.

### POST /api/speech-to-text

Simple transcription without diarization.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (wav, mp3, flac, ogg, m4a, webm) |
| **Limits** | ≤ 100 MB; timeout 5 min |

**Response (200)**

```json
{
  "text": "transcribed text",
  "words": [],
  "languageCode": null
}
```

---

### POST /api/transcribe-diarize

Full pipeline: transcription + speaker diarization + per-segment emotion analysis.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (wav, mp3, flac, ogg, m4a, webm) |
| **Query** | `num_speakers` (optional, integer 1–10) — hint for speaker count; 0 = auto-detect |
| **Limits** | ≤ 100 MB; timeout 10 min |

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
  "text": "full transcript text",
  "filename": "recording.m4a",
  "diarization_method": "vad_mfcc"
}
```

`diarization_method` is either `"pyannote"` (when `HF_TOKEN` is set on the model) or `"vad_mfcc"` (fallback).

---

### GET /health

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

---

## Environment variables

Create `demo/.env.local` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:3000` | Server layer URL used by the browser |

Create `demo/server/.env` or export:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Server layer port |
| `MODEL_URL` | `http://127.0.0.1:8000` | Model layer URL |
