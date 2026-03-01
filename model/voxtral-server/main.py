"""
Evoxtral speech-to-text API proxy (Model layer).
Forwards audio to the external Modal evoxtral API, adds VAD segmentation
and emotion parsing. For video files, also runs FER (MobileViT-XXS ONNX).
"""
import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

EVOXTRAL_API = os.environ.get(
    "EVOXTRAL_API",
    "https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run",
).rstrip("/")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "100")) * 1024 * 1024
TARGET_SR = 16000

# ─── FER setup ────────────────────────────────────────────────────────────────

_fer_session = None
_fer_input_name = "input"
_face_cascade = None
_FER_CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v"}


def _init_fer() -> None:
    global _fer_session, _fer_input_name, _face_cascade

    candidates = [
        os.environ.get("FER_MODEL_PATH", ""),
        "/app/models/emotion_model_web.onnx",
        os.path.join(os.path.dirname(__file__), "../../models/emotion_model_web.onnx"),
    ]
    fer_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not fer_path:
        print("[voxtral] FER model not found — facial emotion disabled")
        return

    try:
        import onnxruntime as rt
        _fer_session = rt.InferenceSession(fer_path, providers=["CPUExecutionProvider"])
        _fer_input_name = _fer_session.get_inputs()[0].name
        print(f"[voxtral] FER model loaded: {fer_path} (input={_fer_input_name})")
    except Exception as e:
        print(f"[voxtral] FER model load failed: {e}")
        return

    try:
        import cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        print("[voxtral] Face cascade loaded")
    except Exception as e:
        print(f"[voxtral] Face cascade load failed (FER will use center crop): {e}")


def _is_video(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in _VIDEO_EXTS


def _fer_frame(img_bgr: np.ndarray) -> Optional[str]:
    """Detect face (or center-crop), run FER ONNX; return emotion label or None."""
    if _fer_session is None:
        return None
    try:
        import cv2
        face_crop = None

        # Try face detection
        if _face_cascade is not None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
                pad = int(min(w, h) * 0.15)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img_bgr.shape[1], x + w + pad), min(img_bgr.shape[0], y + h + pad)
                face_crop = img_bgr[y1:y2, x1:x2]

        if face_crop is None:
            # Center crop fallback (upper 60% of frame = typical head/shoulder area)
            h, w = img_bgr.shape[:2]
            crop_h = int(h * 0.6)
            cx = w // 2
            half = min(crop_h, w) // 2
            face_crop = img_bgr[:crop_h, max(0, cx - half):cx + half]

        resized = cv2.resize(face_crop, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalization (matches original emotion-recognition.ts)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb  = (rgb - mean) / std
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis]  # [1, 3, 224, 224]

        out = _fer_session.run(None, {_fer_input_name: tensor})[0]  # [1, 8]
        return _FER_CLASSES[int(np.argmax(out[0]))]
    except Exception as e:
        print(f"[voxtral] FER frame error: {e}")
        return None


def _fer_for_segments(video_path: str, segments: list[dict]) -> dict[int, str]:
    """Extract ~1fps frames from video, run FER, return {segment_id: emotion}."""
    if _fer_session is None:
        return {}

    frames_dir = tempfile.mkdtemp()
    try:
        import cv2
        from collections import Counter

        rc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "fps=1", "-vframes", "600",
                "-q:v", "5", os.path.join(frames_dir, "%06d.jpg"),
            ],
            capture_output=True, timeout=120,
        )
        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        if not frame_files:
            print("[voxtral] FER: no video frames extracted (audio-only?)")
            return {}

        # second → emotion (000001.jpg = second 0)
        frame_emotions: dict[int, str] = {}
        for fname in frame_files:
            second = int(os.path.splitext(fname)[0]) - 1
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is None:
                continue
            emo = _fer_frame(img)
            if emo:
                frame_emotions[second] = emo

        # Aggregate: most common emotion per VAD segment
        result: dict[int, str] = {}
        for seg in segments:
            start_s, end_s = int(seg["start"]), max(int(seg["end"]), int(seg["start"]) + 1)
            emos = [frame_emotions[s] for s in range(start_s, end_s) if s in frame_emotions]
            if emos:
                result[seg["id"]] = Counter(emos).most_common(1)[0][0]

        print(f"[voxtral] FER: {len(frame_files)} frames → {len(result)} segments with face emotion")
        return result

    except Exception as e:
        print(f"[voxtral] FER extraction error: {e}")
        return {}
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


# ─── Startup ──────────────────────────────────────────────────────────────────

def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found.\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_ffmpeg()
    print(f"[voxtral] ffmpeg: {shutil.which('ffmpeg')}")
    print(f"[voxtral] Evoxtral API: {EVOXTRAL_API}")
    _init_fer()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{EVOXTRAL_API}/health")
            print(f"[voxtral] External API health: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[voxtral] External API health check failed: {e}")
    yield


app = FastAPI(title="Evoxtral Speech-to-Text (Model)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "YongkangZOU/evoxtral-lora (external API)",
        "model_loaded": True,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "fer_enabled": _fer_session is not None,
        "pyannote_available": False,
        "hf_token_set": False,
        "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
        "evoxtral_api": EVOXTRAL_API,
    }


# ─── External STT API ─────────────────────────────────────────────────────────

async def _call_evoxtral(wav_path: str) -> dict:
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{EVOXTRAL_API}/transcribe",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        )
    if not r.is_success:
        raise HTTPException(
            status_code=502,
            detail=f"Evoxtral API error {r.status_code}: {r.text[:300]}",
        )
    return r.json()


# ─── Audio helpers ─────────────────────────────────────────────────────────────

def _convert_to_wav_ffmpeg(path: str, target_sr: int) -> str:
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    rc = subprocess.run(
        ["ffmpeg", "-y", "-i", path,
         "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1",
         "-f", "wav", out.name],
        capture_output=True, timeout=120,
    )
    if rc.returncode != 0:
        os.unlink(out.name)
        raise RuntimeError(f"ffmpeg failed: {rc.stderr.decode(errors='replace')[:400]}")
    return out.name


def _load_audio(file_path: str, target_sr: int) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=target_sr, mono=True)
    return y.astype(np.float32)


def _validate_upload(contents: bytes) -> None:
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)/1024/1024:.1f} MB); max {MAX_UPLOAD_BYTES//1024//1024} MB",
        )


# ─── Segmentation ──────────────────────────────────────────────────────────────

def _vad_segment(audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
    intervals = librosa.effects.split(audio, top_db=28, frame_length=2048, hop_length=512)
    if len(intervals) == 0:
        return [(0, len(audio))]
    merged: list[list[int]] = [[int(intervals[0][0]), int(intervals[0][1])]]
    for s, e in intervals[1:]:
        if (int(s) - merged[-1][1]) / sr < 0.3:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])
    result = [(s, e) for s, e in merged if (e - s) / sr >= 0.3]
    return result if result else [(0, len(audio))]


def _segments_from_vad(audio: np.ndarray, sr: int) -> tuple[list[dict], str]:
    intervals = _vad_segment(audio, sr)
    segs = [
        {"id": i + 1, "speaker": "SPEAKER_00", "start": round(s / sr, 3), "end": round(e / sr, 3)}
        for i, (s, e) in enumerate(intervals)
    ]
    print(f"[voxtral] VAD: {len(segs)} segment(s)")
    return segs, "vad"


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[？！。?!])\s*', text)
    return [p for p in parts if p.strip()]


def _distribute_text(full_text: str, segs: list[dict]) -> list[dict]:
    if not full_text or not segs:
        return [{**s, "text": ""} for s in segs]
    if len(segs) == 1:
        return [{**segs[0], "text": full_text}]
    sentences = _split_sentences(full_text)
    if len(sentences) <= 1:
        is_cjk = len(full_text.split()) <= 1
        sentences = list(full_text) if is_cjk else full_text.split()
    total_dur = sum(s["end"] - s["start"] for s in segs)
    if total_dur <= 0:
        return [{**segs[0], "text": full_text}] + [{**s, "text": ""} for s in segs[1:]]
    is_cjk = len(full_text.split()) <= 1 and len(full_text) > 1
    sep = "" if is_cjk else " "
    n = len(sentences)
    result_texts: list[list[str]] = [[] for _ in segs]
    cumulative = 0.0
    for i, seg in enumerate(segs):
        cumulative += (seg["end"] - seg["start"]) / total_dur
        threshold = cumulative * n
        while len(result_texts[i]) + sum(len(t) for t in result_texts[:i]) < round(threshold):
            idx = sum(len(t) for t in result_texts)
            if idx >= n:
                break
            result_texts[i].append(sentences[idx])
    assigned = sum(len(t) for t in result_texts)
    result_texts[-1].extend(sentences[assigned:])
    return [{**seg, "text": sep.join(texts)} for seg, texts in zip(segs, result_texts)]


# ─── Emotion parsing from evoxtral expression tags ─────────────────────────────

_TAG_EMOTIONS: dict[str, tuple[str, float, float]] = {
    "laughs":           ("Happy",      0.70,  0.60),
    "laughing":         ("Happy",      0.70,  0.60),
    "chuckles":         ("Happy",      0.50,  0.30),
    "giggles":          ("Happy",      0.60,  0.40),
    "sighs":            ("Sad",       -0.30, -0.30),
    "sighing":          ("Sad",       -0.30, -0.30),
    "cries":            ("Sad",       -0.70,  0.40),
    "crying":           ("Sad",       -0.70,  0.40),
    "whispers":         ("Calm",       0.10, -0.50),
    "whispering":       ("Calm",       0.10, -0.50),
    "shouts":           ("Angry",     -0.50,  0.80),
    "shouting":         ("Angry",     -0.50,  0.80),
    "exclaims":         ("Excited",    0.50,  0.70),
    "gasps":            ("Surprised",  0.20,  0.70),
    "hesitates":        ("Anxious",   -0.20,  0.30),
    "stutters":         ("Anxious",   -0.20,  0.40),
    "stammers":         ("Anxious",   -0.25,  0.35),
    "mumbles":          ("Sad",       -0.20, -0.30),
    "nervous":          ("Anxious",   -0.30,  0.40),
    "frustrated":       ("Frustrated",-0.50,  0.50),
    "excited":          ("Excited",    0.50,  0.70),
    "sad":              ("Sad",       -0.60, -0.20),
    "angry":            ("Angry",     -0.60,  0.70),
    "claps":            ("Happy",      0.60,  0.50),
    "applause":         ("Happy",      0.60,  0.50),
    "clears throat":    ("Neutral",    0.00,  0.10),
    "pause":            ("Neutral",    0.00, -0.10),
    "laughs nervously": ("Anxious",   -0.10,  0.40),
}


def _parse_emotion(text: str) -> dict:
    tags = re.findall(r'\[([^\]]+)\]', text.lower())
    for tag in tags:
        tag = tag.strip()
        if tag in _TAG_EMOTIONS:
            label, valence, arousal = _TAG_EMOTIONS[tag]
            return {"emotion": label, "valence": valence, "arousal": arousal}
        for key, (label, valence, arousal) in _TAG_EMOTIONS.items():
            if key in tag:
                return {"emotion": label, "valence": valence, "arousal": arousal}
    return {"emotion": "Neutral", "valence": 0.0, "arousal": 0.0}


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    req_start = time.perf_counter()
    filename = audio.filename or "audio.wav"
    print(f"[voxtral] POST /transcribe filename={filename}")

    try:
        contents = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    _validate_upload(contents)

    suffix = os.path.splitext(filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    wav_path = None
    try:
        wav_path = _convert_to_wav_ffmpeg(tmp_path, TARGET_SR)
        result = await _call_evoxtral(wav_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot process audio: {e}")
    finally:
        for p in (tmp_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    text = result.get("transcription", "")
    print(f"[voxtral] /transcribe done {(time.perf_counter()-req_start)*1000:.0f}ms")
    return {"text": text, "words": [], "languageCode": result.get("language")}


@app.post("/transcribe-diarize")
async def transcribe_diarize(audio: UploadFile = File(...)):
    """
    Upload audio/video → transcription + VAD segmentation + emotion.
    For video files (.mp4, .mkv, .avi, .mov, .m4v), also runs FER
    (MobileViT-XXS ONNX) to add face_emotion per segment.
    """
    req_start = time.perf_counter()
    req_id = f"diarize-{int(req_start * 1000)}"
    filename = audio.filename or "audio.wav"
    print(f"[voxtral] {req_id} POST /transcribe-diarize filename={filename}")

    try:
        contents = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    _validate_upload(contents)

    suffix = os.path.splitext(filename)[1].lower() or ".wav"
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm",
                      ".mp4", ".mkv", ".avi", ".mov", ".m4v"):
        suffix = ".wav"

    # Save upload to disk (needed for both WAV conversion and FER frame extraction)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    wav_path = None
    try:
        # ── Convert to WAV for STT + VAD ────────────────────────────────────
        t0 = time.perf_counter()
        wav_path = _convert_to_wav_ffmpeg(tmp_path, TARGET_SR)
        audio_array = _load_audio(wav_path, TARGET_SR)
        print(f"[voxtral] {req_id} audio loaded shape={audio_array.shape} in {(time.perf_counter()-t0)*1000:.0f}ms")
    except Exception as e:
        for p in (tmp_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        raise HTTPException(status_code=400, detail=f"Cannot decode audio: {e}")

    duration = round(len(audio_array) / TARGET_SR, 3)

    # ── STT via external evoxtral API ────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        result = await _call_evoxtral(wav_path)
        full_text = result.get("transcription", "")
        print(f"[voxtral] {req_id} STT done {(time.perf_counter()-t0)*1000:.0f}ms text_len={len(full_text)}")
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    # ── VAD segmentation + text distribution ─────────────────────────────────
    raw_segs, seg_method = _segments_from_vad(audio_array, TARGET_SR)
    segs_with_text = _distribute_text(full_text, raw_segs)

    # ── FER (video files only, runs in thread pool to avoid blocking) ─────────
    has_fer = False
    face_emotions: dict[int, str] = {}
    if _is_video(filename) and _fer_session is not None:
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        face_emotions = await loop.run_in_executor(None, _fer_for_segments, tmp_path, raw_segs)
        has_fer = bool(face_emotions)
        print(f"[voxtral] {req_id} FER done {(time.perf_counter()-t0)*1000:.0f}ms faces={len(face_emotions)}")

    # Clean up original upload
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Build segments ────────────────────────────────────────────────────────
    segments = []
    for s in segs_with_text:
        emo = _parse_emotion(s["text"])
        seg_data = {
            "id":      s["id"],
            "speaker": s["speaker"],
            "start":   s["start"],
            "end":     s["end"],
            "text":    s["text"],
            "emotion": emo["emotion"],
            "valence": emo["valence"],
            "arousal": emo["arousal"],
        }
        if s["id"] in face_emotions:
            seg_data["face_emotion"] = face_emotions[s["id"]]
        segments.append(seg_data)

    total_ms = (time.perf_counter() - req_start) * 1000
    print(f"[voxtral] {req_id} complete total={total_ms:.0f}ms segments={len(segments)} has_fer={has_fer}")

    return {
        "segments": segments,
        "duration": duration,
        "text": full_text,
        "filename": filename,
        "diarization_method": seg_method,
        "has_video": has_fer,
    }
