# Ser - frontend & backend

This folder contains the Next.js frontend and Node proxy backend.

## Frontend (Next.js)

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

First, run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Speech to Text (Backend-Only)

Speech-to-text service based on [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).

### Architecture

```
Client
  → Server layer (Node, port 3000)   POST /api/speech-to-text, GET /health
      → Model layer (Python, port 8000)   POST /transcribe, GET /health
```

- **Model layer** (`model/voxtral-server`): Voxtral inference. Exposes `POST /transcribe`, `GET /health`. See [model/voxtral-server/README.md](../model/voxtral-server/README.md) for API details.
- **Server layer** (`demo/server`): Single entrypoint for clients. Proxies to Model layer. Exposes `POST /api/speech-to-text`, `GET /health`. See [demo/server/README.md](server/README.md) for API details.

---

### Startup

#### 1. Model layer (Python, port 8000)

Requires **Python 3.10+**, **ffmpeg**, and (recommended) GPU ≥16GB VRAM.

```bash
cd ../model/voxtral-server
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

First run may download the model (~8–16GB). Wait until you see `Application startup complete` and `Uvicorn running on http://0.0.0.0:8000`.

#### 2. Server layer (Node, port 3001)

Wait, the backend proxy runs on 3000 by default. It may conflict with Next.js running on 3000. 

```bash
cd server
npm install
PORT=3001 npm run dev
```

You should see `Server layer listening on http://0.0.0.0:3001`.

#### 3. Quick check

```bash
curl -s http://localhost:3001/health
```

Expect `{"ok":true,"server":"ser-server","model":{...}}` when both layers are up.

---

### API usage (Server layer)

Clients should call the **Server layer** only. The Model layer is used internally by the server.

#### POST /api/speech-to-text

Transcribe an audio file.

| | |
|--|--|
| **Request** | `POST /api/speech-to-text` |
| **Content-Type** | `multipart/form-data` |
| **Body** | One field: `audio` = audio file (binary) |
| **Limits** | File size ≤ 100 MB (configurable via Server code), timeout 5 min |

**Success (200)**  
JSON body:

```json
{
  "text": "transcribed text here",
  "words": [],
  "languageCode": null
}
```
