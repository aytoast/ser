# Speech to Text (Backend-Only)

Speech-to-text service based on [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602). **Backend only**: no frontend — **Model layer** (Python) and **Server layer** (Node) only.

## Architecture

```
Client
  → Server layer (Node, port 3000)   POST /api/speech-to-text, GET /health
      → Model layer (Python, port 8000)   POST /transcribe, GET /health
```

- **Model layer** (`model/voxtral-server`): Voxtral inference. Exposes `POST /transcribe`, `GET /health`. See [model/voxtral-server/README.md](../model/voxtral-server/README.md) for API details.
- **Server layer** (`demo/server`): Single entrypoint for clients. Proxies to Model layer. Exposes `POST /api/speech-to-text`, `GET /health`. See [demo/server/README.md](server/README.md) for API details.

---

## Startup

### 1. Model layer (Python, port 8000)

Requires **Python 3.10+**, **ffmpeg**, and (recommended) GPU ≥16GB VRAM.

```bash
cd model/voxtral-server
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

First run may download the model (~8–16GB). Wait until you see `Application startup complete` and `Uvicorn running on http://0.0.0.0:8000`.

### 2. Server layer (Node, port 3000)

Requires **Node.js 18+**.

```bash
cd demo/server
npm install
npm run dev
```

You should see `Server layer listening on http://0.0.0.0:3000`.

### 3. Quick check

```bash
curl -s http://localhost:3000/health
```

Expect `{"ok":true,"server":"ser-server","model":{...}}` when both layers are up.

---

## API usage (Server layer)

Clients should call the **Server layer** only. The Model layer is used internally by the server.

### POST /api/speech-to-text

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

**Errors**

| Status | Meaning |
|--------|--------|
| 400 | Missing or invalid `audio` field, or file too large |
| 502 | Model layer error or unreachable |
| 504 | Transcription timeout (over 5 min) |

### GET /health

Health check (and Model layer status).

**Success (200)**  
e.g. `{"ok":true,"server":"ser-server","model":{"status":"ok","model":"mistralai/...",...}}`

**Errors**  
502 if Model layer is unreachable.

---

## Model layer (called by Server)

The Server forwards transcription to the Model layer. For direct testing or integration you can call it yourself:

- **Base URL**: `http://127.0.0.1:8000` (default)
- **POST /transcribe**: `multipart/form-data`, field `audio` = audio file. Returns `{"text":"...","words":[],"languageCode":null}`.
- **GET /health**: Returns `{"status":"ok","model":"...","model_loaded":true,"ffmpeg":"...","max_upload_mb":100}`.

Supported formats: wav, mp3, flac, ogg, m4a, webm. See [model/voxtral-server/README.md](../model/voxtral-server/README.md) for full API and parameters.

---

## Usage examples

### cURL — transcribe a file

```bash
curl -X POST http://localhost:3000/api/speech-to-text \
  -F "audio=@/path/to/audio.m4a"
```

Example response:

```json
{"text":"Hello world","words":[],"languageCode":null}
```

### cURL — health check

```bash
curl -s http://localhost:3000/health | jq
```

### Python

```python
import requests

url = "http://localhost:3000/api/speech-to-text"
with open("/path/to/audio.m4a", "rb") as f:
    r = requests.post(url, files={"audio": ("audio.m4a", f, "audio/mp4")})
print(r.json())  # {"text": "...", "words": [], "languageCode": null}
```

### Node (fetch)

```js
const form = new FormData();
form.append("audio", new Blob([await fs.readFile("/path/to/audio.m4a")]), "audio.m4a");
const r = await fetch("http://localhost:3000/api/speech-to-text", { method: "POST", body: form });
console.log(await r.json());
```

---

## Environment variables

| Layer | Variable | Default | Description |
|-------|----------|---------|-------------|
| Server | `PORT` | `3000` | Server listen port |
| Server | `MODEL_URL` | `http://127.0.0.1:8000` | Model layer base URL |
| Model | `VOXTRAL_MODEL_ID` | `mistralai/Voxtral-Mini-4B-Realtime-2602` | Hugging Face model ID |
| Model | `MAX_UPLOAD_MB` | `100` | Max upload size in MB |

Optional: create `.env` in `demo/server` (and load with dotenv if you add it) to set `MODEL_URL` and `PORT`.
