/**
 * Server layer: proxy client requests to Model layer (voxtral-server).
 * Port default 3000, Model layer default http://127.0.0.1:8000
 */
import express from "express";
import multer from "multer";
import cors from "cors";

const PORT = Number(process.env.PORT) || 3000;
const MODEL_URL = (process.env.MODEL_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100 MB
const TRANSCRIBE_TIMEOUT_MS = 30 * 60 * 1000;   // 30 min (CPU inference is slow)
const DIARIZE_TIMEOUT_MS   = 60 * 60 * 1000;    // 60 min (CPU: ~50s audio/min)

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES },
});

const app = express();

app.use(cors({
  origin: [
    "http://localhost:3030",
    "http://127.0.0.1:3030",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
  ],
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"],
}));

app.use((req, res, next) => {
  const start = Date.now();
  res.on("finish", () => {
    console.log("[server]", req.method, req.path, res.statusCode, `${Date.now() - start}ms`);
  });
  next();
});

// ─── /health ──────────────────────────────────────────────────────────────────
app.get("/health", async (req, res) => {
  try {
    const r = await fetch(`${MODEL_URL}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      return res.status(502).json({ ok: false, error: "Model layer unavailable", detail: data });
    }
    res.json({ ok: true, server: "ser-server", model: data });
  } catch (err) {
    console.error("[server] health check model:", err?.message || err);
    res.status(502).json({
      ok: false,
      error: "Cannot reach Model layer; start model/voxtral-server first",
      url: MODEL_URL,
    });
  }
});

// ─── shared proxy helper ──────────────────────────────────────────────────────
async function proxyToModel(req, res, modelPath, timeoutMs) {
  const reqId = `req-${Date.now().toString(36)}`;
  const start = Date.now();

  if (!req.file) {
    return res.status(400).json({ error: "Upload an audio file (form field: audio)" });
  }

  const { buffer, size, originalname } = req.file;
  if (size > MAX_UPLOAD_BYTES) {
    return res.status(400).json({
      error: `File size exceeds ${MAX_UPLOAD_BYTES / 1024 / 1024}MB limit`,
    });
  }

  const form = new FormData();
  form.append("audio", new Blob([buffer]), originalname || "audio");

  // Forward num_speakers query param if present
  const numSpeakers = req.query.num_speakers;
  const url = numSpeakers
    ? `${MODEL_URL}${modelPath}?num_speakers=${encodeURIComponent(numSpeakers)}`
    : `${MODEL_URL}${modelPath}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    console.log(`[server] ${reqId} → ${url} file=${originalname} size=${size}`);
    const r = await fetch(url, { method: "POST", body: form, signal: controller.signal });
    clearTimeout(timeoutId);

    const rawText = await r.text().catch(() => "");
    let data = {};
    try { data = JSON.parse(rawText); } catch {}

    if (!r.ok) {
      const errMsg = data.detail || data.error || "Failed";
      console.error(`[server] ${reqId} model error ${r.status}: ${errMsg} | raw=${rawText.slice(0, 300)}`);
      return res.status(r.status >= 500 ? 502 : r.status).json({
        error: typeof errMsg === "string" ? errMsg : "Model error",
      });
    }

    console.log(`[server] ${reqId} ok in ${Date.now() - start}ms`);
    res.json(data);
  } catch (err) {
    clearTimeout(timeoutId);
    const isAbort = err.name === "AbortError";
    console.error(`[server] ${reqId} ${isAbort ? "timeout" : "error"} after ${Date.now() - start}ms:`, err.message);
    res.status(isAbort ? 504 : 502).json({
      error: isAbort
        ? `Request timeout (>${timeoutMs / 60000} min); try shorter audio`
        : "Cannot reach Model layer; ensure voxtral-server is running",
    });
  }
}

// ─── /api/debug-inference (proxies to model /debug-inference) ────────────────
app.get("/api/debug-inference", async (req, res) => {
  try {
    const r = await fetch(`${MODEL_URL}/debug-inference`, { signal: AbortSignal.timeout(60000) });
    const data = await r.json().catch(() => ({ error: "non-JSON response from model" }));
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

// ─── /api/speech-to-text ──────────────────────────────────────────────────────
app.post("/api/speech-to-text", upload.single("audio"), (req, res) => {
  return proxyToModel(req, res, "/transcribe", TRANSCRIBE_TIMEOUT_MS);
});

// ─── /api/transcribe-diarize ──────────────────────────────────────────────────
app.post("/api/transcribe-diarize", upload.single("audio"), (req, res) => {
  return proxyToModel(req, res, "/transcribe-diarize", DIARIZE_TIMEOUT_MS);
});

// ─── start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[server] Server layer listening on http://0.0.0.0:${PORT}`);
  console.log("[server] Model layer URL:", MODEL_URL);
  console.log("[server] POST /api/speech-to-text        → batch transcription");
  console.log("[server] POST /api/transcribe-diarize    → transcription + speaker diarization");
});
