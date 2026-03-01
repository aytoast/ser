/**
 * Server layer: proxy client requests to Model layer (voxtral-server).
 * Port default 3000, Model layer default http://127.0.0.1:8000
 */
import express from "express";
import multer from "multer";

const PORT = Number(process.env.PORT) || 3000;
const MODEL_URL = (process.env.MODEL_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100 MB
const TRANSCRIBE_TIMEOUT_MS = 5 * 60 * 1000; // 5 min

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES },
});

const app = express();

app.use((req, res, next) => {
  const start = Date.now();
  req._start = start;
  res.on("finish", () => {
    console.log(
      "[server]",
      req.method,
      req.path,
      res.statusCode,
      `${Date.now() - start}ms`
    );
  });
  next();
});

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

app.post(
  "/api/speech-to-text",
  upload.single("audio"),
  async (req, res) => {
    const start = Date.now();
    const reqId = `req-${Date.now().toString(36)}`;
    console.log("[server]", reqId, "POST /api/speech-to-text received");

    if (!req.file) {
      console.warn("[server]", reqId, "400 missing audio or invalid form field");
      return res.status(400).json({ error: "Upload an audio file (form field: audio)" });
    }

    const { buffer, size, originalname } = req.file;
    console.log("[server]", reqId, "file received:", originalname, "size:", size, "bytes");
    if (size > MAX_UPLOAD_BYTES) {
      console.warn("[server]", reqId, "400 file too large", originalname, `${(size / 1024 / 1024).toFixed(2)}MB`);
      return res.status(400).json({
        error: `File size exceeds ${MAX_UPLOAD_BYTES / 1024 / 1024}MB limit`,
      });
    }

    console.log("[server]", reqId, "building FormData and calling Model layer", `${MODEL_URL}/transcribe`);
    const form = new FormData();
    form.append("audio", new Blob([buffer]), originalname || "audio");

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TRANSCRIBE_TIMEOUT_MS);

    try {
      const fetchStart = Date.now();
      console.log("[server]", reqId, "fetch started at", new Date().toISOString());
      const r = await fetch(`${MODEL_URL}/transcribe`, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      const fetchElapsed = Date.now() - fetchStart;
      clearTimeout(timeoutId);
      console.log("[server]", reqId, "fetch completed:", r.status, "in", fetchElapsed + "ms");

      const elapsed = Date.now() - start;
      const data = await r.json().catch((parseErr) => {
        console.error("[server]", reqId, "response JSON parse error:", parseErr?.message || parseErr);
        return {};
      });

      if (!r.ok) {
        const errMsg = data.detail || data.error || "Transcription failed";
        console.error("[server]", reqId, "Model error", r.status, "elapsed", elapsed + "ms", "detail:", errMsg);
        return res
          .status(r.status >= 500 ? 502 : r.status)
          .json({ error: typeof errMsg === "string" ? errMsg : "Transcription failed" });
      }

      const textLen = typeof data.text === "string" ? data.text.length : 0;
      console.log("[server]", reqId, "200 transcription ok total", elapsed + "ms", "text length:", textLen);
      res.json(data);
    } catch (err) {
      clearTimeout(timeoutId);
      const elapsed = Date.now() - start;
      const isAbort = err.name === "AbortError";
      console.error(
        "[server]", reqId,
        isAbort ? "504 timeout" : "502 request failed",
        "elapsed", elapsed + "ms",
        "name:", err.name,
        "message:", err.message || err
      );
      res
        .status(isAbort ? 504 : 502)
        .json({
          error: isAbort
            ? "Transcription timeout (over 5 min); try shorter audio or retry later"
            : "Cannot reach Model layer; ensure voxtral-server is running",
        });
    }
  }
);

app.listen(PORT, () => {
  console.log(`[server] Server layer listening on http://0.0.0.0:${PORT}`);
  console.log("[server] Model layer URL:", MODEL_URL);
  console.log("[server] Transcribe: POST /api/speech-to-text (form-data audio)");
});
