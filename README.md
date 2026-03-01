# Ethos Studio — Emotional Speech Recognition

Speech-to-text service with VAD sentence segmentation and per-segment emotion analysis, powered by [Voxtral Mini 4B](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602). Three-layer architecture: **Model** (Python) + **Server** (Node) + **Frontend** (Next.js).

## Architecture

```
Browser (port 3030)  →  Server layer (Node, :3000)  →  Model layer (Python, :8000)
      ↑ Studio UI            POST /api/speech-to-text          POST /transcribe
      ↑ Upload dialog        POST /api/transcribe-diarize      POST /transcribe-diarize
                             GET  /health                       GET  /health
```

| Layer | Path | Role |
|-------|------|------|
| **Model** | `model/voxtral-server` | Voxtral inference, VAD sentence segmentation, emotion analysis |
| **Server** | `demo/server` | API entrypoint; proxies to Model |
| **Frontend** | `demo` | Next.js UI (upload, Studio editor, waveform, timeline) |

See [demo/README.md](demo/README.md) for full API and usage; [model/voxtral-server/README.md](model/voxtral-server/README.md) for the Model API.

## How to run

**Requirements**: Python 3.10+, Node.js 20+, ffmpeg; GPU ≥16GB VRAM recommended (Apple Silicon MPS supported).

### 1. Start Model layer (port 8000)

```bash
cd model/voxtral-server
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for `Application startup complete`. First run may download the model (~8–16GB).

### 2. Start Server layer (port 3000)

```bash
cd demo/server
npm install
npm run dev
```

### 3. Start Frontend (port 3030)

```bash
cd demo
npm install
npm run dev
```

Open [http://localhost:3030](http://localhost:3030).

### 4. Quick check

```bash
curl -s http://localhost:3000/health
curl -X POST http://localhost:3000/api/speech-to-text -F "audio=@/path/to/audio.m4a"
curl -X POST http://localhost:3000/api/transcribe-diarize -F "audio=@/path/to/audio.m4a"
```
