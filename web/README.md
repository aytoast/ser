# Frontend (Next.js — port 3030)

Ethos Studio UI. Upload audio or video files, view transcription results with per-segment emotion badges and facial emotion (FER) badges, and explore the waveform timeline in the Studio editor.

---

## Architecture

```
Browser (port 3030)
  → Proxy layer (Node, port 3000)   POST /api/speech-to-text, POST /api/transcribe-diarize, GET /health
      → API layer (Python, port 8000)   POST /transcribe, POST /transcribe-diarize, GET /health
```

- **Frontend** (`web/`): Upload page + Studio editor. Calls the Proxy layer.
- **Proxy layer** (`proxy/`): Forwards browser requests to the API layer. See [proxy/README.md](../proxy/README.md) for API details.
- **API layer** (`api/`): Local Voxtral inference + VAD segmentation + emotion + FER. See [api/README.md](../api/README.md) for API details.

---

## Startup

### 1. API layer (Python, port 8000)

Requires **Python 3.11+** and **ffmpeg**.

```bash
cd api
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On first run the Voxtral model (~8 GB) is downloaded from HuggingFace.

### 2. Proxy layer (Node, port 3000)

```bash
cd proxy
npm install
npm run dev
```

### 3. Frontend (Next.js, port 3030)

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3030](http://localhost:3030).

- **Home page**: Click **Transcribe files**, drag-drop an audio or video file, then **Upload**. The file is sent to `/api/transcribe-diarize` and results open in the Studio.
- **Studio page** (`/studio`): Three-column layout — transcript segments (speaker + emotion badges + FER badges for video) on the left, waveform in the center, audio player on the right.

### 4. Quick check (API only)

```bash
curl -s http://localhost:3000/health
curl -X POST http://localhost:3000/api/speech-to-text -F "audio=@audio.m4a"
curl -X POST http://localhost:3000/api/transcribe-diarize -F "audio=@audio.m4a"
curl -X POST http://localhost:3000/api/transcribe-diarize -F "audio=@video.mov"
```

---

## API (Proxy layer)

Clients should call the **Proxy layer** only. The API layer is internal.

### POST /api/speech-to-text

Simple transcription without diarization.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio file (wav, mp3, flac, ogg, m4a, webm) |
| **Limits** | ≤ 100 MB; timeout 30 min |

**Response (200)**

```json
{
  "text": "transcribed text",
  "words": [],
  "languageCode": "en"
}
```

---

### POST /api/transcribe-diarize

Full pipeline: transcription + VAD sentence segmentation + per-segment emotion analysis. For video inputs, also returns `face_emotion` per segment.

| | |
|--|--|
| **Content-Type** | `multipart/form-data` |
| **Body** | `audio` — audio or video file (wav, mp3, flac, ogg, m4a, webm, mp4, mov, mkv) |
| **Limits** | ≤ 100 MB; timeout 60 min |

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
  "text": "full transcript text",
  "filename": "recording.mov",
  "diarization_method": "vad",
  "has_video": true
}
```

`face_emotion` appears only on video uploads when FER is enabled. `diarization_method` is always `"vad"`.

---

### GET /health

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

---

## Environment variables

Create `web/.env.local`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:3000` | Proxy layer URL used by the browser |

Create `proxy/.env` or export:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Proxy layer port |
| `MODEL_URL` | `http://127.0.0.1:8000` | API layer URL |
