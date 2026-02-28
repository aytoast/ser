"""
Voxtral speech-to-text API (offline transcription) - Model layer.
Model ID can be overridden with env VOXTRAL_MODEL_ID; default mistralai/Voxtral-Mini-4B-Realtime-2602
"""
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager

import torch
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REPO_ID = os.environ.get("VOXTRAL_MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "100")) * 1024 * 1024

processor = None
model = None


def _check_ffmpeg():
    """Check ffmpeg is available at startup; raise with clear message if not."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. WebM (e.g. browser recording) requires ffmpeg to decode.\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html\n"
            "Then restart this service."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: check deps and load model."""
    global processor, model

    # 1. Check ffmpeg (required for WebM)
    _check_ffmpeg()
    print(f"[voxtral] ffmpeg: {shutil.which('ffmpeg')}")

    # 2. Load model (offline: AutoProcessor + VoxtralRealtimeForConditionalGeneration)
    print(f"[voxtral] Loading model: {REPO_ID} (first run may download ~8–16GB)...")
    try:
        from transformers import (
            VoxtralRealtimeForConditionalGeneration,
            AutoProcessor,
        )
        processor = AutoProcessor.from_pretrained(REPO_ID)
        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            REPO_ID, device_map="auto", torch_dtype=torch.bfloat16
        )
        model.eval()
        print(f"[voxtral] Model loaded: {REPO_ID}")
    except Exception as e:
        raise RuntimeError(
            f"Model load failed: {e}\n"
            "Ensure deps are installed: pip install -r requirements.txt\n"
            "And sufficient VRAM (recommended ≥16GB) or use CPU (slower)."
        ) from e

    yield


app = FastAPI(title="Voxtral Speech-to-Text (Model)", lifespan=lifespan)

# CORS: allow Server layer (Node) to call
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check: service and dependency status."""
    return {
        "status": "ok",
        "model": REPO_ID,
        "model_loaded": model is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
    }


def _convert_to_wav_ffmpeg(path: str, target_sr: int) -> str:
    """Convert any format to 16kHz mono WAV with ffmpeg; return path to new file."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    rc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", path,
            "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1",
            "-f", "wav", out.name,
        ],
        capture_output=True,
        timeout=120,
    )
    if rc.returncode != 0:
        os.unlink(out.name)
        raise RuntimeError(
            f"ffmpeg failed: {rc.stderr.decode(errors='replace')[:500]}"
        )
    return out.name


def load_audio_to_array(file_path: str, target_sr: int) -> np.ndarray:
    """Load audio to mono float32 and resample to target_sr.
    WebM/Opus/M4A etc. use ffmpeg first to avoid PySoundFile/audioread hangs.
    Only .wav/.mp3/.flac try librosa first, then ffmpeg fallback.
    """
    lower = file_path.lower()
    # Problematic formats: use ffmpeg to avoid PySoundFile/audioread issues
    if lower.endswith((".webm", ".opus", ".m4a", ".ogg")):
        wav_path = _convert_to_wav_ffmpeg(file_path, target_sr)
        try:
            y, _ = librosa.load(wav_path, sr=target_sr, mono=True)
            return y.astype(np.float32)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    # Only wav/mp3/flac try librosa first
    try:
        y, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return y.astype(np.float32)
    except Exception as e:
        if not os.path.isfile(file_path):
            raise
        need_ffmpeg = (
            "format not recognised" in str(e).lower()
            or "nobackenderror" in str(type(e).__name__).lower()
        )
        if need_ffmpeg:
            wav_path = _convert_to_wav_ffmpeg(file_path, target_sr)
            try:
                y, _ = librosa.load(wav_path, sr=target_sr, mono=True)
                return y.astype(np.float32)
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
        raise


def _validate_upload(contents: bytes) -> None:
    """Validate upload: non-empty and within size limit."""
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty; record at least 1–2 seconds or choose a valid file")
    if len(contents) > MAX_UPLOAD_BYTES:
        mb = len(contents) / 1024 / 1024
        limit_mb = MAX_UPLOAD_BYTES // 1024 // 1024
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({mb:.1f} MB); max {limit_mb} MB",
        )


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Upload an audio file; return full transcription (offline, single response).
    Supported: wav, mp3, flac, ogg, m4a, webm
    """
    req_start = time.perf_counter()
    req_id = f"transcribe-{int(req_start * 1000)}"
    filename = audio.filename or "audio.wav"
    print(f"[voxtral] {req_id} POST /transcribe received filename={filename}")

    try:
        contents = await audio.read()
    except Exception as e:
        print(f"[voxtral] {req_id} read file error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    print(f"[voxtral] {req_id} read {len(contents)} bytes in {(time.perf_counter() - req_start)*1000:.0f}ms")

    _validate_upload(contents)

    suffix = os.path.splitext(filename)[1].lower() or ".wav"
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"):
        suffix = ".wav"

    target_sr = processor.feature_extractor.sampling_rate
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    print(f"[voxtral] {req_id} wrote temp file {tmp_path}")

    try:
        t0 = time.perf_counter()
        audio_array = load_audio_to_array(tmp_path, target_sr)
        print(f"[voxtral] {req_id} load_audio_to_array done shape={audio_array.shape} in {(time.perf_counter()-t0)*1000:.0f}ms")
    except Exception as e:
        print(f"[voxtral] {req_id} load_audio_to_array error: {e}")
        raise HTTPException(status_code=400, detail=f"Cannot decode audio: {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    t0 = time.perf_counter()
    with torch.no_grad():
        inputs = processor(audio_array, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)
        print(f"[voxtral] {req_id} processor done in {(time.perf_counter()-t0)*1000:.0f}ms")
        t1 = time.perf_counter()
        outputs = model.generate(**{k: v for k, v in inputs.items()})
        print(f"[voxtral] {req_id} model.generate done in {(time.perf_counter()-t1)*1000:.0f}ms")
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)

    text = (decoded[0] or "").strip()
    total_ms = (time.perf_counter() - req_start) * 1000
    print(f"[voxtral] {req_id} done total={total_ms:.0f}ms text_len={len(text)}")
    return {"text": text, "words": [], "languageCode": None}
