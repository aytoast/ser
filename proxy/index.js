/**
 * Server layer: proxy client requests to Model layer (voxtral-server).
 * Port default 3000, Model layer default http://127.0.0.1:8000
 *
 * POST /api/transcribe-diarize → returns {job_id} immediately (202)
 * GET  /api/job/:id            → returns {status, data?, error?}
 * Polling avoids HF Spaces ~3 min proxy timeout during long CPU inference.
 */
import express from "express";
import multer from "multer";
import cors from "cors";

const PORT = Number(process.env.PORT) || 3000;
const MODEL_URL = (process.env.MODEL_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100 MB
const DIARIZE_TIMEOUT_MS = 60 * 60 * 1000;  // 60 min (CPU: ~50s/min of audio)
const JOB_TTL_MS = 30 * 60 * 1000;          // keep completed jobs 30 min then evict

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES },
});

const app = express();

app.use(cors({
  origin: "*",
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

// ─── Job store ────────────────────────────────────────────────────────────────
/** @type {Map<string, {status:"pending"|"done"|"error", data?:object, error?:string, ts:number}>} */
const jobs = new Map();

function evictOldJobs() {
  const cutoff = Date.now() - JOB_TTL_MS;
  for (const [id, job] of jobs) {
    if (job.status !== "pending" && job.ts < cutoff) jobs.delete(id);
  }
}
setInterval(evictOldJobs, 5 * 60 * 1000);

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

// ─── Background job processor ─────────────────────────────────────────────────
async function runDiarizeJob(jobId, file, query) {
  const reqId = `req-${Date.now().toString(36)}`;
  const start = Date.now();
  const { buffer, size, originalname } = file;

  const form = new FormData();
  form.append("audio", new Blob([buffer]), originalname || "audio");

  const numSpeakers = query?.num_speakers;
  const url = numSpeakers
    ? `${MODEL_URL}/transcribe-diarize?num_speakers=${encodeURIComponent(numSpeakers)}`
    : `${MODEL_URL}/transcribe-diarize`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DIARIZE_TIMEOUT_MS);

  try {
    console.log(`[server] ${reqId} job=${jobId} → ${url} file=${originalname} size=${size}`);
    const r = await fetch(url, { method: "POST", body: form, signal: controller.signal });
    clearTimeout(timeoutId);

    const rawText = await r.text().catch(() => "");
    let data = {};
    try { data = JSON.parse(rawText); } catch {}

    if (!r.ok) {
      const errMsg = data.detail || data.error || "Failed";
      console.error(`[server] ${reqId} model error ${r.status}: ${errMsg}`);
      jobs.set(jobId, { status: "error", error: typeof errMsg === "string" ? errMsg : "Model error", ts: Date.now() });
      return;
    }

    console.log(`[server] ${reqId} job=${jobId} done in ${Date.now() - start}ms`);
    jobs.set(jobId, { status: "done", data, ts: Date.now() });
  } catch (err) {
    clearTimeout(timeoutId);
    const isAbort = err.name === "AbortError";
    console.error(`[server] ${reqId} job=${jobId} ${isAbort ? "timeout" : "error"} after ${Date.now() - start}ms:`, err.message);
    jobs.set(jobId, {
      status: "error",
      error: isAbort
        ? `Request timeout (>60 min); try shorter audio`
        : "Cannot reach Model layer; ensure voxtral-server is running",
      ts: Date.now(),
    });
  }
}

// ─── /api/job/:id — poll for job result ───────────────────────────────────────
app.get("/api/job/:id", (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: "Job not found or expired" });
  if (job.status === "pending") return res.json({ status: "pending" });
  if (job.status === "error") return res.status(200).json({ status: "error", error: job.error });
  return res.json({ status: "done", data: job.data });
});

// ─── /api/transcribe-diarize — submit job, return immediately ─────────────────
app.post("/api/transcribe-diarize", upload.single("audio"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Upload an audio file (form field: audio)" });
  }
  if (req.file.size > MAX_UPLOAD_BYTES) {
    return res.status(400).json({ error: `File size exceeds ${MAX_UPLOAD_BYTES / 1024 / 1024}MB limit` });
  }

  const jobId = `job-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
  jobs.set(jobId, { status: "pending", ts: Date.now() });

  // Respond immediately — don't await
  res.status(202).json({ job_id: jobId });

  // Kick off background processing
  runDiarizeJob(jobId, req.file, req.query).catch(err => {
    jobs.set(jobId, { status: "error", error: err.message, ts: Date.now() });
  });
});

// ─── /api/debug-inference ─────────────────────────────────────────────────────
app.get("/api/debug-inference", async (req, res) => {
  try {
    const r = await fetch(`${MODEL_URL}/debug-inference`, { signal: AbortSignal.timeout(60000) });
    const data = await r.json().catch(() => ({ error: "non-JSON response from model" }));
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

// ─── start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[server] Server layer listening on http://0.0.0.0:${PORT}`);
  console.log("[server] Model layer URL:", MODEL_URL);
  console.log("[server] POST /api/transcribe-diarize  → submit async job (202 + job_id)");
  console.log("[server] GET  /api/job/:id             → poll job status");
});
