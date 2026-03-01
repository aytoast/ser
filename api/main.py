"""
Ethos Studio — API layer
STT via Mistral Voxtral API (cloud, fast), emotion via Mistral chat (batch),
FER (facial emotion) via local MobileViT-XXS ONNX.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral

# ─── Config ───────────────────────────────────────────────────────────────────

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
STT_MODEL  = os.environ.get("STT_MODEL",  "voxtral-mini-latest")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "mistral-small-latest")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "100")) * 1024 * 1024
TARGET_SR = 16000

_mistral: Optional[Mistral] = None

# ─── FER setup ────────────────────────────────────────────────────────────────

_fer_session    = None
_fer_input_name = "input"
_face_cascade   = None
_FER_CLASSES    = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
_VIDEO_EXTS     = {".mp4", ".mkv", ".avi", ".mov", ".m4v"}


def _init_fer() -> None:
    global _fer_session, _fer_input_name, _face_cascade

    candidates = [
        os.environ.get("FER_MODEL_PATH", ""),
        "/app/models/emotion_model_web.onnx",
        os.path.join(os.path.dirname(__file__), "../models/emotion_model_web.onnx"),
        os.path.join(os.path.dirname(__file__), "../../models/emotion_model_web.onnx"),
    ]
    fer_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not fer_path:
        print("[api] FER model not found — facial emotion disabled")
        return

    try:
        import onnxruntime as rt
        _fer_session    = rt.InferenceSession(fer_path, providers=["CPUExecutionProvider"])
        _fer_input_name = _fer_session.get_inputs()[0].name
        print(f"[api] FER model loaded: {fer_path}")
    except Exception as e:
        print(f"[api] FER model load failed: {e}")
        return

    try:
        import cv2
        cascade_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        print("[api] Face cascade loaded")
    except Exception as e:
        print(f"[api] Face cascade load failed (will use center crop): {e}")


def _is_video(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in _VIDEO_EXTS


def _fer_frame(img_bgr: np.ndarray) -> Optional[str]:
    if _fer_session is None:
        return None
    try:
        import cv2
        face_crop = None

        if _face_cascade is not None:
            gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad = int(min(w, h) * 0.15)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img_bgr.shape[1], x + w + pad), min(img_bgr.shape[0], y + h + pad)
                face_crop = img_bgr[y1:y2, x1:x2]

        if face_crop is None:
            h, w   = img_bgr.shape[:2]
            crop_h = int(h * 0.6)
            cx     = w // 2
            half   = min(crop_h, w) // 2
            face_crop = img_bgr[:crop_h, max(0, cx - half):cx + half]

        resized = cv2.resize(face_crop, (224, 224))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb     = (rgb - mean) / std
        tensor  = np.transpose(rgb, (2, 0, 1))[np.newaxis]

        out = _fer_session.run(None, {_fer_input_name: tensor})[0]
        return _FER_CLASSES[int(np.argmax(out[0]))]
    except Exception as e:
        print(f"[api] FER frame error: {e}")
        return None


def _fer_for_segments(video_path: str, segments: list[dict]) -> dict[int, str]:
    if _fer_session is None:
        return {}
    frames_dir = tempfile.mkdtemp()
    try:
        import cv2
        from collections import Counter

        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", "fps=1", "-vframes", "600",
             "-q:v", "5", os.path.join(frames_dir, "%06d.jpg")],
            capture_output=True, timeout=120,
        )
        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        if not frame_files:
            return {}

        frame_emotions: dict[int, str] = {}
        for fname in frame_files:
            second = int(os.path.splitext(fname)[0]) - 1
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is None:
                continue
            emo = _fer_frame(img)
            if emo:
                frame_emotions[second] = emo

        result: dict[int, str] = {}
        for seg in segments:
            start_s = int(seg["start"])
            end_s   = max(int(seg["end"]), start_s + 1)
            emos    = [frame_emotions[s] for s in range(start_s, end_s) if s in frame_emotions]
            if emos:
                result[seg["id"]] = Counter(emos).most_common(1)[0][0]

        print(f"[api] FER: {len(frame_files)} frames → {len(result)} segments with face emotion")
        return result
    except Exception as e:
        print(f"[api] FER extraction error: {e}")
        return {}
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


# ─── Startup ──────────────────────────────────────────────────────────────────

def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg / apt install ffmpeg")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mistral
    _check_ffmpeg()
    _init_fer()
    if MISTRAL_API_KEY:
        _mistral = Mistral(api_key=MISTRAL_API_KEY)
        print(f"[api] Mistral client ready — STT: {STT_MODEL}  chat: {CHAT_MODEL}")
    else:
        print("[api] WARNING: MISTRAL_API_KEY not set — transcription will fail")
    yield


app = FastAPI(title="Ethos Studio API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─── Health / debug ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": f"{STT_MODEL} (Mistral API)",
        "model_loaded": _mistral is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "fer_enabled": _fer_session is not None,
        "device": "cloud",
        "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
    }


@app.get("/debug-inference")
async def debug_inference():
    """Smoke-test: send 1s of silence to the Voxtral API."""
    if _mistral is None:
        return {"ok": False, "error": "MISTRAL_API_KEY not set"}
    try:
        import struct, wave, io
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        silence_bytes = buf.getvalue()

        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: _mistral.audio.transcriptions.complete(
                model=STT_MODEL,
                file={"content": silence_bytes, "file_name": "silence.wav"},
                language="en",
            )
        )
        return {"ok": True, "text": resp.text, "model": STT_MODEL}
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


# ─── Audio helpers ────────────────────────────────────────────────────────────

def _validate_upload(contents: bytes) -> None:
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)/1024/1024:.1f} MB); max {MAX_UPLOAD_BYTES//1024//1024} MB",
        )


def _extract_audio_ffmpeg(input_path: str, target_sr: int) -> str:
    """Extract/convert audio to WAV (strips video track)."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    rc = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1",
         "-f", "wav", out.name],
        capture_output=True, timeout=120,
    )
    if rc.returncode != 0:
        os.unlink(out.name)
        raise RuntimeError(f"ffmpeg failed: {rc.stderr.decode(errors='replace')[:400]}")
    return out.name


def _audio_duration_ffmpeg(wav_path: str) -> float:
    rc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
        capture_output=True, timeout=30,
    )
    try:
        return float(rc.stdout.decode().strip())
    except Exception:
        return 0.0


# ─── STT via Mistral API ──────────────────────────────────────────────────────

def _call_voxtral_api(audio_bytes: bytes, filename: str):
    """Blocking call to Mistral Voxtral transcription API."""
    return _mistral.audio.transcriptions.complete(
        model=STT_MODEL,
        file={"content": audio_bytes, "file_name": filename},
        language="en",
        diarize=True,
        timestamp_granularities=["segment"],
    )


def _parse_api_segments(resp, duration: float) -> tuple[list[dict], str]:
    """Convert Mistral API response to our segment format."""
    raw = getattr(resp, "segments", None) or []
    if raw:
        segs = []
        for i, s in enumerate(raw):
            segs.append({
                "id": i + 1,
                "speaker": getattr(s, "speaker", None) or f"SPEAKER_{i:02d}",
                "start": round(float(getattr(s, "start", 0)), 3),
                "end":   round(float(getattr(s, "end",   0)), 3),
                "text":  (getattr(s, "text", "") or "").strip(),
            })
        return segs, "api"
    # Fallback: single segment covering full duration
    text = (getattr(resp, "text", "") or "").strip()
    return [{"id": 1, "speaker": "SPEAKER_00", "start": 0.0, "end": duration, "text": text}], "vad"


# ─── Emotion via Mistral chat ─────────────────────────────────────────────────

_EMOTION_CHOICES = "Happy, Sad, Angry, Excited, Calm, Anxious, Surprised, Neutral, Frustrated"

def _analyze_emotions_sync(texts: list[str]) -> list[dict]:
    """One Mistral chat call → emotion + valence + arousal for all segments."""
    if not texts or _mistral is None:
        return [{"emotion": "Neutral", "valence": 0.0, "arousal": 0.0}] * len(texts)

    prompt = (
        f"Classify the speaker emotion for each speech segment below.\n"
        f"Choose emotion from: {_EMOTION_CHOICES}.\n"
        f"Valence: -1.0 (very negative) to 1.0 (very positive).\n"
        f"Arousal: -1.0 (calm/low energy) to 1.0 (excited/high energy).\n"
        f"Return ONLY a JSON array — one object per segment, same order:\n"
        f'[{{"emotion":"...","valence":0.0,"arousal":0.0}}, ...]\n\n'
        f"Segments:\n{json.dumps(texts, ensure_ascii=False)}"
    )

    try:
        resp = _mistral.chat.complete(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content or ""
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            emotions = json.loads(match.group())
            result = []
            for i in range(len(texts)):
                emo = emotions[i] if i < len(emotions) else {}
                result.append({
                    "emotion": str(emo.get("emotion", "Neutral")),
                    "valence": float(emo.get("valence", 0.0)),
                    "arousal": float(emo.get("arousal", 0.0)),
                })
            return result
    except Exception as e:
        print(f"[api] emotion analysis error: {e}")

    return [{"emotion": "Neutral", "valence": 0.0, "arousal": 0.0}] * len(texts)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if _mistral is None:
        raise HTTPException(status_code=503, detail="MISTRAL_API_KEY not configured")

    t0 = time.perf_counter()
    filename = audio.filename or "audio.wav"
    contents = await audio.read()
    _validate_upload(contents)

    # For video files extract audio first
    suffix = os.path.splitext(filename)[1].lower() or ".wav"
    if suffix in _VIDEO_EXTS:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            wav_path = _extract_audio_ffmpeg(tmp_path, TARGET_SR)
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
            audio_filename = "audio.wav"
        finally:
            for p in (tmp_path,):
                if os.path.exists(p): os.unlink(p)
            if 'wav_path' in locals() and os.path.exists(wav_path): os.unlink(wav_path)
    else:
        audio_bytes = contents
        audio_filename = filename

    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(None, _call_voxtral_api, audio_bytes, audio_filename)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mistral API error: {e}")

    print(f"[api] /transcribe done {(time.perf_counter()-t0)*1000:.0f}ms")
    return {"text": resp.text or "", "words": []}


@app.post("/transcribe-diarize")
async def transcribe_diarize(audio: UploadFile = File(...)):
    if _mistral is None:
        raise HTTPException(status_code=503, detail="MISTRAL_API_KEY not configured")

    req_start = time.perf_counter()
    req_id    = f"diarize-{int(req_start*1000)}"
    filename  = audio.filename or "audio.wav"
    print(f"[api] {req_id} POST /transcribe-diarize filename={filename}")

    contents = await audio.read()
    _validate_upload(contents)

    suffix = os.path.splitext(filename)[1].lower() or ".wav"
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm",
                      ".mp4", ".mkv", ".avi", ".mov", ".m4v"):
        suffix = ".wav"

    # Save original file (needed for FER on video)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    wav_path = None
    try:
        # Extract audio WAV for API call
        t0 = time.perf_counter()
        wav_path = _extract_audio_ffmpeg(tmp_path, TARGET_SR)
        duration = _audio_duration_ffmpeg(wav_path)
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        print(f"[api] {req_id} audio extracted duration={duration:.1f}s in {(time.perf_counter()-t0)*1000:.0f}ms")

        # ── STT via Mistral API ───────────────────────────────────────────────
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, _call_voxtral_api, audio_bytes, "audio.wav")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Mistral API error: {e}")

        full_text = (resp.text or "").strip()
        segs, seg_method = _parse_api_segments(resp, duration)
        print(f"[api] {req_id} STT done {(time.perf_counter()-t0)*1000:.0f}ms "
              f"text_len={len(full_text)} segments={len(segs)} method={seg_method}")

        # ── Emotion via Mistral chat (batch) ─────────────────────────────────
        t0 = time.perf_counter()
        texts = [s["text"] for s in segs]
        emotions = await loop.run_in_executor(None, _analyze_emotions_sync, texts)
        print(f"[api] {req_id} emotions done {(time.perf_counter()-t0)*1000:.0f}ms")

        # ── FER (video only) ──────────────────────────────────────────────────
        face_emotions: dict[int, str] = {}
        has_fer = False
        if _is_video(filename) and _fer_session is not None:
            t0 = time.perf_counter()
            face_emotions = await loop.run_in_executor(None, _fer_for_segments, tmp_path, segs)
            has_fer = bool(face_emotions)
            print(f"[api] {req_id} FER done {(time.perf_counter()-t0)*1000:.0f}ms faces={len(face_emotions)}")

    finally:
        for p in filter(None, [tmp_path, wav_path]):
            if os.path.exists(p):
                try: os.unlink(p)
                except OSError: pass

    # ── Build response ────────────────────────────────────────────────────────
    segments = []
    for seg, emo in zip(segs, emotions):
        seg_data = {
            "id":      seg["id"],
            "speaker": seg["speaker"],
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    seg["text"],
            "emotion": emo["emotion"],
            "valence": emo["valence"],
            "arousal": emo["arousal"],
        }
        if seg["id"] in face_emotions:
            seg_data["face_emotion"] = face_emotions[seg["id"]]
        segments.append(seg_data)

    total_ms = (time.perf_counter() - req_start) * 1000
    print(f"[api] {req_id} complete total={total_ms:.0f}ms segments={len(segments)}")

    return {
        "segments":           segments,
        "duration":           duration,
        "text":               full_text,
        "filename":           filename,
        "diarization_method": seg_method,
        "has_video":          has_fer,
    }
