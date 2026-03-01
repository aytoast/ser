"""
Evoxtral speech-to-text API proxy (Model layer).
Forwards audio to the external Modal evoxtral API, then adds
VAD segmentation and emotion parsing from inline expression tags.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager

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


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. WebM / M4A / OGG requires ffmpeg to decode.\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_ffmpeg()
    print(f"[voxtral] ffmpeg: {shutil.which('ffmpeg')}")
    print(f"[voxtral] Evoxtral API: {EVOXTRAL_API}")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{EVOXTRAL_API}/health")
            print(f"[voxtral] External API health: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[voxtral] External API health check failed: {e} (will retry on first request)")
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
        "pyannote_available": False,
        "hf_token_set": False,
        "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
        "evoxtral_api": EVOXTRAL_API,
    }


# ─── External API ──────────────────────────────────────────────────────────────

async def _call_evoxtral(contents: bytes, filename: str) -> dict:
    """Forward audio bytes to the external evoxtral API; return parsed JSON.
    Response: {"transcription": "...[laughs]...", "language": "en", "model": "..."}
    """
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{EVOXTRAL_API}/transcribe",
            files={"file": (filename, contents)},
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
        raise RuntimeError(f"ffmpeg failed: {rc.stderr.decode(errors='replace')[:500]}")
    return out.name


def _load_audio(file_path: str, target_sr: int) -> np.ndarray:
    lower = file_path.lower()
    if lower.endswith((".webm", ".opus", ".m4a", ".ogg")):
        wav_path = _convert_to_wav_ffmpeg(file_path, target_sr)
        try:
            y, _ = librosa.load(wav_path, sr=target_sr, mono=True)
            return y.astype(np.float32)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
    try:
        y, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return y.astype(np.float32)
    except Exception as e:
        need_ffmpeg = (
            "format not recognised" in str(e).lower()
            or "nobackenderror" in type(e).__name__.lower()
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
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty; record at least 1–2 seconds or choose a valid file",
        )
    if len(contents) > MAX_UPLOAD_BYTES:
        mb = len(contents) / 1024 / 1024
        limit_mb = MAX_UPLOAD_BYTES // 1024 // 1024
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({mb:.1f} MB); max {limit_mb} MB",
        )


# ─── Segmentation helpers ──────────────────────────────────────────────────────

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
        {"speaker": "SPEAKER_00", "start": round(s / sr, 3), "end": round(e / sr, 3)}
        for s, e in intervals
    ]
    print(f"[voxtral] VAD segmentation: {len(segs)} segment(s)")
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

# Maps inline tags like [laughs], [sighs] → (emotion label, valence, arousal)
_TAG_EMOTIONS: dict[str, tuple[str, float, float]] = {
    "laughs":    ("Happy",     0.70,  0.60),
    "laughing":  ("Happy",     0.70,  0.60),
    "chuckles":  ("Happy",     0.50,  0.30),
    "giggles":   ("Happy",     0.60,  0.40),
    "sighs":     ("Sad",      -0.30, -0.30),
    "sighing":   ("Sad",      -0.30, -0.30),
    "cries":     ("Sad",      -0.70,  0.40),
    "crying":    ("Sad",      -0.70,  0.40),
    "whispers":  ("Calm",      0.10, -0.50),
    "whispering":("Calm",      0.10, -0.50),
    "shouts":    ("Angry",    -0.50,  0.80),
    "shouting":  ("Angry",    -0.50,  0.80),
    "exclaims":  ("Excited",   0.50,  0.70),
    "gasps":     ("Surprised", 0.20,  0.70),
    "hesitates": ("Anxious",  -0.20,  0.30),
    "stutters":  ("Anxious",  -0.20,  0.40),
    "mumbles":   ("Sad",      -0.20, -0.30),
    "claps":     ("Happy",     0.60,  0.50),
    "applause":  ("Happy",     0.60,  0.50),
}


def _parse_emotion(text: str) -> dict:
    """Extract the first recognized expression tag from text like [sighs] or [laughs].
    Returns {"emotion": str, "valence": float, "arousal": float}.
    Defaults to Neutral (0, 0) if no known tag is found.
    """
    tags = re.findall(r'\[([^\]]+)\]', text.lower())
    for tag in tags:
        tag = tag.strip()
        if tag in _TAG_EMOTIONS:
            label, valence, arousal = _TAG_EMOTIONS[tag]
            return {"emotion": label, "valence": valence, "arousal": arousal}
        # Partial match (e.g. "laughs softly" → "laughs")
        for key, (label, valence, arousal) in _TAG_EMOTIONS.items():
            if key in tag:
                return {"emotion": label, "valence": valence, "arousal": arousal}
    return {"emotion": "Neutral", "valence": 0.0, "arousal": 0.0}


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Upload audio → plain transcription (with inline expression tags)."""
    req_start = time.perf_counter()
    req_id = f"transcribe-{int(req_start * 1000)}"
    filename = audio.filename or "audio.wav"
    print(f"[voxtral] {req_id} POST /transcribe filename={filename}")

    try:
        contents = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    _validate_upload(contents)

    result = await _call_evoxtral(contents, filename)
    text = result.get("transcription", "")
    lang = result.get("language")

    total_ms = (time.perf_counter() - req_start) * 1000
    print(f"[voxtral] {req_id} done total={total_ms:.0f}ms text_len={len(text)}")
    return {"text": text, "words": [], "languageCode": lang}


@app.post("/transcribe-diarize")
async def transcribe_diarize(audio: UploadFile = File(...)):
    """
    Upload audio → transcription + VAD segmentation + per-segment emotion.
    Transcription is produced by the external evoxtral API (includes expressive tags).
    Emotion is parsed from inline tags like [sighs], [laughs], etc.
    All segments are labelled SPEAKER_00 (single-speaker mode).
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
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"):
        suffix = ".wav"

    # ── Step 1: call external evoxtral API ──────────────────────────────────
    t0 = time.perf_counter()
    result = await _call_evoxtral(contents, filename)
    full_text = result.get("transcription", "")
    print(f"[voxtral] {req_id} evoxtral API done {(time.perf_counter()-t0)*1000:.0f}ms text_len={len(full_text)}")

    # ── Step 2: load audio for VAD segmentation ──────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        t0 = time.perf_counter()
        audio_array = _load_audio(tmp_path, TARGET_SR)
        print(f"[voxtral] {req_id} load_audio done shape={audio_array.shape} in {(time.perf_counter()-t0)*1000:.0f}ms")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode audio: {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    duration = round(len(audio_array) / TARGET_SR, 3)

    # ── Step 3: VAD sentence segmentation ───────────────────────────────────
    t0 = time.perf_counter()
    raw_segs, seg_method = _segments_from_vad(audio_array, TARGET_SR)
    print(f"[voxtral] {req_id} segmentation done {(time.perf_counter()-t0)*1000:.0f}ms segs={len(raw_segs)}")

    # ── Step 4: distribute text to segments ─────────────────────────────────
    segs_with_text = _distribute_text(full_text, raw_segs)

    # ── Step 5: parse emotion from expression tags ──────────────────────────
    segments = []
    for i, s in enumerate(segs_with_text):
        emo = _parse_emotion(s["text"])
        segments.append({
            "id":      i + 1,
            "speaker": s["speaker"],
            "start":   s["start"],
            "end":     s["end"],
            "text":    s["text"],
            "emotion": emo["emotion"],
            "valence": emo["valence"],
            "arousal": emo["arousal"],
        })

    total_ms = (time.perf_counter() - req_start) * 1000
    print(f"[voxtral] {req_id} complete total={total_ms:.0f}ms segments={len(segments)}")

    return {
        "segments": segments,
        "duration": duration,
        "text": full_text,
        "filename": filename,
        "diarization_method": seg_method,
    }
