"""
Voxtral speech-to-text API (offline transcription + speaker diarization) - Model layer.
Model ID can be overridden with env VOXTRAL_MODEL_ID; default mistralai/Voxtral-Mini-4B-Realtime-2602
"""
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

REPO_ID = os.environ.get("VOXTRAL_MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "100")) * 1024 * 1024
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional: enables pyannote speaker diarization

processor = None
model = None

# Optional: pyannote pipeline (loaded lazily on first diarize request if HF_TOKEN is set)
_pyannote_pipeline = None
_pyannote_loaded = False
_pyannote_available = False

try:
    from pyannote.audio import Pipeline as _PyannotePipeline
    _pyannote_available = True
except ImportError:
    pass


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


def _get_pyannote_pipeline():
    """Lazy-load pyannote pipeline (requires HF_TOKEN and pyannote.audio installed)."""
    global _pyannote_pipeline, _pyannote_loaded
    if _pyannote_loaded:
        return _pyannote_pipeline
    _pyannote_loaded = True
    if not _pyannote_available or not HF_TOKEN:
        print("[voxtral] pyannote: not available (install pyannote.audio and set HF_TOKEN for real diarization; using VAD+MFCC fallback)")
        return None
    try:
        pipeline = _PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
        _pyannote_pipeline = pipeline
        print("[voxtral] pyannote speaker-diarization-3.1 loaded")
    except Exception as e:
        print(f"[voxtral] pyannote load failed: {e} — using VAD+MFCC fallback")
    return _pyannote_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: check deps and load model."""
    global processor, model

    _check_ffmpeg()
    print(f"[voxtral] ffmpeg: {shutil.which('ffmpeg')}")

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
        "pyannote_available": _pyannote_available,
        "hf_token_set": bool(HF_TOKEN),
        "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
    }


# ─── Audio helpers ─────────────────────────────────────────────────────────────

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
    """Load audio to mono float32 and resample to target_sr."""
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


# ─── Speaker diarization helpers ───────────────────────────────────────────────

def _vad_split(audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Split audio on silence, merge gaps < 0.8 s, filter segments < 0.5 s.
    Returns list of (start_sample, end_sample) tuples.
    """
    intervals = librosa.effects.split(audio, top_db=28, frame_length=2048, hop_length=512)
    if len(intervals) == 0:
        return [(0, len(audio))]

    # Merge intervals with gap < 0.8 s
    merged: list[list[int]] = [[int(intervals[0][0]), int(intervals[0][1])]]
    for s, e in intervals[1:]:
        if (int(s) - merged[-1][1]) / sr < 0.8:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])

    # Filter very short segments
    result = [(s, e) for s, e in merged if (e - s) / sr >= 0.4]
    return result if result else [(0, len(audio))]


def _extract_mfcc_features(segments: list[tuple[int, int]], audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract normalised MFCC feature matrix (one row per segment)."""
    from sklearn.preprocessing import StandardScaler

    rows = []
    for s, e in segments:
        chunk = audio[s:e]
        if len(chunk) < 512:
            rows.append(np.zeros(40))
            continue
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
        rows.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    X = np.array(rows)
    return StandardScaler().fit_transform(X)


def _auto_num_speakers(X: np.ndarray, max_speakers: int = 8) -> int:
    """Pick the number of speakers that maximises silhouette score (k=2..max_k)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    max_k = min(max_speakers, len(X))
    if max_k < 2:
        return 1

    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels))
        print(f"[voxtral]   silhouette k={k}: {score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"[voxtral] auto-detected {best_k} speaker(s) (silhouette={best_score:.4f})")
    return best_k


def _cluster_speakers(
    segments: list[tuple[int, int]],
    audio: np.ndarray,
    sr: int,
    n_speakers: int,          # 0 = auto-detect
) -> list[int]:
    """Assign speaker IDs to segments via MFCC + KMeans.
    Pass n_speakers=0 to automatically detect the speaker count.
    Falls back to alternating labels if sklearn unavailable or clustering fails.
    """
    if len(segments) <= 1:
        return [0] * len(segments)

    try:
        from sklearn.cluster import KMeans

        X = _extract_mfcc_features(segments, audio, sr)

        if n_speakers == 0:
            n_speakers = _auto_num_speakers(X)

        n_speakers = min(n_speakers, len(segments))
        labels = KMeans(n_clusters=n_speakers, random_state=42, n_init=10).fit_predict(X)
        return [int(l) for l in labels]
    except Exception as e:
        print(f"[voxtral] MFCC clustering failed: {e} — using alternating labels")
        k = max(1, n_speakers)
        return [i % k for i in range(len(segments))]


def _segments_from_pyannote(wav_path: str) -> Optional[list[dict]]:
    """Run pyannote pipeline and return raw segments. Returns None if unavailable."""
    pipeline = _get_pyannote_pipeline()
    if pipeline is None:
        return None
    try:
        diarization = pipeline(wav_path)
        segs = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segs.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            })
        return segs if segs else None
    except Exception as e:
        print(f"[voxtral] pyannote inference failed: {e}")
        return None


def _segments_from_vad(audio: np.ndarray, sr: int, n_speakers: int) -> list[dict]:
    """Fallback: VAD split + MFCC speaker clustering."""
    intervals = _vad_split(audio, sr)
    labels = _cluster_speakers(intervals, audio, sr, n_speakers)
    segs = []
    for (s, e), spk in zip(intervals, labels):
        segs.append({
            "speaker": f"SPEAKER_{spk:02d}",
            "start": round(s / sr, 3),
            "end": round(e / sr, 3),
        })
    return segs


def _distribute_text(full_text: str, segs: list[dict]) -> list[dict]:
    """Proportionally distribute transcription words across segments by duration."""
    words = full_text.split()
    total_words = len(words)
    if not words or not segs:
        return [{**s, "text": ""} for s in segs]

    total_dur = sum(s["end"] - s["start"] for s in segs)
    if total_dur <= 0:
        result = [{**segs[0], "text": full_text}]
        return result + [{**s, "text": ""} for s in segs[1:]]

    result: list[dict] = []
    word_idx = 0
    for i, seg in enumerate(segs):
        dur = seg["end"] - seg["start"]
        frac = dur / total_dur
        n = round(frac * total_words)
        if i == len(segs) - 1:
            chunk = words[word_idx:]
        else:
            chunk = words[word_idx: word_idx + max(1, n)]
        result.append({**seg, "text": " ".join(chunk)})
        word_idx += len(chunk)
    return result


# ─── Emotion analysis ──────────────────────────────────────────────────────────

def _emotion_label(valence: float, arousal: float) -> str:
    """Map continuous valence/arousal to a discrete emotion label."""
    if arousal > 0.3:
        if valence > 0.15:
            return "Happy" if arousal > 0.6 else "Excited"
        elif valence < -0.15:
            return "Angry" if arousal > 0.6 else "Anxious"
        return "Alert"
    elif arousal < -0.2:
        if valence > 0.15:
            return "Calm"
        elif valence < -0.15:
            return "Sad"
        return "Bored"
    else:
        if valence > 0.2:
            return "Content"
        elif valence < -0.2:
            return "Frustrated"
        return "Neutral"


def _analyze_emotion(chunk: np.ndarray, sr: int) -> dict:
    """Estimate valence/arousal from acoustic features; return {emotion, valence, arousal}.

    Correlates used:
      Arousal  ← RMS energy, mean pitch, zero-crossing rate
      Valence  ← spectral brightness, pitch variation (tonal variety)
    """
    if len(chunk) < 512:
        return {"emotion": "Neutral", "valence": 0.0, "arousal": 0.0}

    try:
        # ── Energy ──────────────────────────────────────────────────────────
        rms = float(librosa.feature.rms(y=chunk).mean())

        # ── Pitch (YIN) ─────────────────────────────────────────────────────
        f0 = librosa.yin(chunk, fmin=60, fmax=450, sr=sr)
        voiced = f0[(f0 > 60) & (f0 < 450)]
        pitch_mean = float(voiced.mean()) if len(voiced) > 0 else 150.0
        pitch_std  = float(voiced.std())  if len(voiced) > 0 else 0.0

        # ── Spectral features ────────────────────────────────────────────────
        spec_centroid = float(librosa.feature.spectral_centroid(y=chunk, sr=sr).mean())
        zcr           = float(librosa.feature.zero_crossing_rate(chunk).mean())

        # ── Arousal (0..1 before rescaling) ─────────────────────────────────
        rms_n   = min(rms / 0.08, 1.0)                    # typical speech RMS
        pitch_n = max(0.0, min((pitch_mean - 80) / 320, 1.0))  # 80–400 Hz
        zcr_n   = min(zcr / 0.12, 1.0)
        arousal_01 = 0.5 * rms_n + 0.35 * pitch_n + 0.15 * zcr_n
        arousal = round(arousal_01 * 2 - 1, 3)            # → -1..1

        # ── Valence (0..1 before rescaling) ─────────────────────────────────
        spec_n     = min(spec_centroid / 3500, 1.0)        # brighter = warmer
        pitch_var_n = min(pitch_std / 60, 1.0)             # melodic variety
        valence_01 = 0.55 * spec_n + 0.45 * pitch_var_n
        valence = round(valence_01 * 2 - 1, 3)            # → -1..1

        emotion = _emotion_label(valence, arousal)
        return {"emotion": emotion, "valence": valence, "arousal": arousal}

    except Exception as e:
        print(f"[voxtral] _analyze_emotion failed: {e}")
        return {"emotion": "Neutral", "valence": 0.0, "arousal": 0.0}


# ─── Endpoints ─────────────────────────────────────────────────────────────────

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
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    _validate_upload(contents)

    suffix = os.path.splitext(filename)[1].lower() or ".wav"
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"):
        suffix = ".wav"

    target_sr = processor.feature_extractor.sampling_rate
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        audio_array = load_audio_to_array(tmp_path, target_sr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode audio: {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    with torch.no_grad():
        inputs = processor(audio_array, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)
        outputs = model.generate(**{k: v for k, v in inputs.items()})
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)

    text = (decoded[0] or "").strip()
    total_ms = (time.perf_counter() - req_start) * 1000
    print(f"[voxtral] {req_id} done total={total_ms:.0f}ms text_len={len(text)}")
    return {"text": text, "words": [], "languageCode": None}


@app.post("/transcribe-diarize")
async def transcribe_diarize(
    audio: UploadFile = File(...),
    num_speakers: int = Query(default=0, ge=0, le=10, description="Expected number of speakers; 0 = auto-detect"),
):
    """
    Upload audio → full transcription + speaker diarization.
    Returns structured segments: [{id, speaker, start, end, text, emotion, valence, arousal}]

    Speaker detection method (in priority order):
      1. pyannote/speaker-diarization-3.1  (needs HF_TOKEN + pyannote.audio installed)
      2. VAD silence split + MFCC KMeans clustering  (zero extra deps, always available)
    """
    req_start = time.perf_counter()
    req_id = f"diarize-{int(req_start * 1000)}"
    filename = audio.filename or "audio.wav"
    print(f"[voxtral] {req_id} POST /transcribe-diarize filename={filename} num_speakers={num_speakers}")

    try:
        contents = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    _validate_upload(contents)

    suffix = os.path.splitext(filename)[1].lower() or ".wav"
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"):
        suffix = ".wav"

    target_sr = processor.feature_extractor.sampling_rate

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        audio_array = load_audio_to_array(tmp_path, target_sr)
        print(f"[voxtral] {req_id} load_audio done shape={audio_array.shape} in {(time.perf_counter()-t0)*1000:.0f}ms")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode audio: {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    duration = round(len(audio_array) / target_sr, 3)

    # ── Step 1: full transcription via Voxtral ──────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        inputs = processor(audio_array, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=model.dtype)
        outputs = model.generate(**{k: v for k, v in inputs.items()})
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    full_text = (decoded[0] or "").strip()
    print(f"[voxtral] {req_id} transcription done in {(time.perf_counter()-t0)*1000:.0f}ms text_len={len(full_text)}")

    # ── Step 2: speaker diarization ─────────────────────────────────────────
    t0 = time.perf_counter()
    raw_segs: Optional[list[dict]] = None

    # Try pyannote first (requires HF_TOKEN)
    if _pyannote_available and HF_TOKEN:
        wav_tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_array, target_sr)
                wav_tmp = f.name
            raw_segs = _segments_from_pyannote(wav_tmp)
        except Exception as e:
            print(f"[voxtral] {req_id} pyannote error: {e}")
        finally:
            if wav_tmp and os.path.exists(wav_tmp):
                os.unlink(wav_tmp)
        if raw_segs:
            print(f"[voxtral] {req_id} pyannote diarization done in {(time.perf_counter()-t0)*1000:.0f}ms segs={len(raw_segs)}")

    # Fallback: VAD + MFCC clustering
    if not raw_segs:
        raw_segs = _segments_from_vad(audio_array, target_sr, num_speakers)
        print(f"[voxtral] {req_id} VAD+MFCC diarization done in {(time.perf_counter()-t0)*1000:.0f}ms segs={len(raw_segs)}")

    # ── Step 3: distribute text proportionally ──────────────────────────────
    segs_with_text = _distribute_text(full_text, raw_segs)

    # ── Step 4: emotion analysis per segment ────────────────────────────────
    t0 = time.perf_counter()
    segments = []
    for i, s in enumerate(segs_with_text):
        start_sample = int(s["start"] * target_sr)
        end_sample   = int(s["end"]   * target_sr)
        chunk = audio_array[start_sample:end_sample]
        emo = _analyze_emotion(chunk, target_sr)
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
    print(f"[voxtral] {req_id} emotion analysis done in {(time.perf_counter()-t0)*1000:.0f}ms")

    total_ms = (time.perf_counter() - req_start) * 1000
    print(f"[voxtral] {req_id} complete total={total_ms:.0f}ms segments={len(segments)}")

    return {
        "segments": segments,
        "duration": duration,
        "text": full_text,
        "filename": filename,
        "diarization_method": "pyannote" if (raw_segs and _pyannote_available and HF_TOKEN) else "vad_mfcc",
    }
