The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/pages/building-your-application/deploying) for more details.


# SER — Speech-to-Text

Backend-only speech-to-text service using [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602). No frontend: **Model layer** (Python) + **Server layer** (Node).

## Architecture

```
Client (curl / script / app)
    → Server layer (Node, :3000)   POST /api/speech-to-text, GET /health
        → Model layer (Python, :8000)   POST /transcribe, GET /health
```

| Layer | Path | Role |
|-------|------|------|
| **Model** | `model/voxtral-server` | Voxtral inference; `POST /transcribe`, `GET /health` |
| **Server** | `demo/server` | API entrypoint; proxies to Model; `POST /api/speech-to-text`, `GET /health` |

See [demo/README.md](demo/README.md) for full API and usage; [model/voxtral-server/README.md](model/voxtral-server/README.md) for Model API.

## How to run

**Requirements**: Python 3.10+, Node.js 18+, ffmpeg; GPU ≥16GB VRAM recommended for Model.

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

### 3. Check
  
```bash
curl -s http://localhost:3000/health
curl -X POST http://localhost:3000/api/speech-to-text -F "audio=@/path/to/audio.m4a"
```
