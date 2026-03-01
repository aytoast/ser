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

REPO_ID = os.environ.get("VOXTRAL_MODEL_ID", "YongkangZOU/evoxtral-lora")
BASE_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
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
        elif torch.backends.mps.is_available():
            pipeline = pipeline.to(torch.device("mps"))
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

    if torch.cuda.is_available():
        _device = torch.device("cuda")
        _dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
        _dtype = torch.float16   # MPS does not support bfloat16
    else:
        _device = torch.device("cpu")
        _dtype = torch.bfloat16  # halves memory vs float32 (8 GB vs 16 GB); supported on modern x86
    print(f"[voxtral] Device: {_device}  dtype: {_dtype}")

    print(f"[voxtral] Loading base model: {BASE_MODEL_ID} ...")
    print(f"[voxtral] Applying LoRA adapter: {REPO_ID} ...")
    try:
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        base = VoxtralForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID, torch_dtype=_dtype
        ).to(_device)
        model = PeftModel.from_pretrained(base, REPO_ID)
        model.eval()
        print(f"[voxtral] Model ready: {BASE_MODEL_ID} + LoRA {REPO_ID} on {_device}")
    except Exception as e:
        raise RuntimeError(
            f"Model load failed: {e}\n"
            "Ensure deps are installed: pip install -r requirements.txt\n"
            "And sufficient VRAM (recommended ≥16GB) or use CPU (slower)."
        ) from e

    # Warm-up: run one silent dummy inference to pre-compile MPS Metal shaders.
    print("[voxtral] Warming up (dummy inference)...")
    try:
        sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
        dummy = np.zeros(sr, dtype=np.float32)  # 1 second of silence
        conversation = [{"role": "user", "content": [{"type": "audio", "audio": dummy}]}]
        with torch.inference_mode():
            dummy_inputs = processor.apply_chat_template(
                conversation, return_tensors="pt", tokenize=True
            ).to(_device)
            model.generate(**dummy_inputs, max_new_tokens=1)
        print("[voxtral] Warm-up complete — first request will be fast")
    except Exception as e:
        print(f"[voxtral] Warm-up skipped: {e}")

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


# ─── Segmentation helpers ──────────────────────────────────────────────────────

def _vad_segment(audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Split audio into speech segments by silence detection.
    Merges gaps < 0.5 s (intra-phrase pauses) and drops segments < 0.3 s.
    Returns list of (start_sample, end_sample).
    """
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
    """Segment audio by silence, assign all segments to SPEAKER_00.
    Returns (segments, method_name).
    """
    intervals = _vad_segment(audio, sr)
    segs = [
        {"speaker": "SPEAKER_00", "start": round(s / sr, 3), "end": round(e / sr, 3)}
        for s, e in intervals
    ]
    print(f"[voxtral] VAD segmentation: {len(segs)} segment(s)")
    return segs, "vad"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at punctuation boundaries (CJK + Latin)."""
    import re
    parts = re.split(r'(?<=[？！。?!])\s*', text)
    return [p for p in parts if p.strip()]


def _distribute_text(full_text: str, segs: list[dict]) -> list[dict]:
    """Assign complete sentences to segments by time proportion.
    Sentences are never split mid-punctuation; each segment gets whole sentences.
    Falls back to character-level splitting if no sentence boundaries found.
    """
    if not full_text or not segs:
        return [{**s, "text": ""} for s in segs]

    if len(segs) == 1:
        return [{**segs[0], "text": full_text}]

    sentences = _split_sentences(full_text)
    # Fallback: split by character if no sentence boundaries
    if len(sentences) <= 1:
        is_cjk = len(full_text.split()) <= 1
        sentences = list(full_text) if is_cjk else full_text.split()

    total_dur = sum(s["end"] - s["start"] for s in segs)
    if total_dur <= 0:
        return [{**segs[0], "text": full_text}] + [{**s, "text": ""} for s in segs[1:]]

    is_cjk = len(full_text.split()) <= 1 and len(full_text) > 1
    sep = "" if is_cjk else " "

    # Assign each sentence to the segment whose cumulative time covers its proportional position
    n = len(sentences)
    result_texts: list[list[str]] = [[] for _ in segs]

    cumulative = 0.0
    for i, seg in enumerate(segs):
        cumulative += (seg["end"] - seg["start"]) / total_dur
        # Assign sentences whose proportional position falls within this segment's cumulative range
        threshold = cumulative * n
        while len(result_texts[i]) + sum(len(t) for t in result_texts[:i]) < round(threshold):
            idx = sum(len(t) for t in result_texts)
            if idx >= n:
                break
            result_texts[i].append(sentences[idx])

    # Ensure any leftover sentences go to the last segment
    assigned = sum(len(t) for t in result_texts)
    result_texts[-1].extend(sentences[assigned:])

    return [{**seg, "text": sep.join(texts)} for seg, texts in zip(segs, result_texts)]


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


# ─── Inference helper ──────────────────────────────────────────────────────────

def _transcribe(audio_array: np.ndarray) -> str:
    """Run Voxtral-3B + LoRA inference via chat template; return transcribed text."""
    import traceback
    audio_sec = round(len(audio_array) / 16000, 2)
    model_dtype = next(model.parameters()).dtype
    print(f"[_transcribe] START audio={audio_sec}s device={model.device} dtype={model_dtype}", flush=True)

    try:
        t0 = time.perf_counter()
        conversation = [{"role": "user", "content": [{"type": "audio", "audio": audio_array}]}]
        inputs = processor.apply_chat_template(
            conversation, return_tensors="pt", tokenize=True, add_generation_prompt=True
        )
        print(f"[_transcribe] apply_chat_template OK {(time.perf_counter()-t0)*1000:.0f}ms keys={list(inputs.keys())}", flush=True)
    except Exception:
        print(f"[_transcribe] apply_chat_template FAILED:\n{traceback.format_exc()}", flush=True)
        raise

    try:
        t0 = time.perf_counter()
        # move to device; cast floating tensors to model dtype to avoid dtype mismatch
        inputs = {
            k: (v.to(model.device, dtype=model_dtype) if v.is_floating_point() else v.to(model.device))
            for k, v in inputs.items()
        }
        input_len = inputs["input_ids"].shape[1]
        print(f"[_transcribe] to(device) OK {(time.perf_counter()-t0)*1000:.0f}ms input_len={input_len}", flush=True)
    except Exception:
        print(f"[_transcribe] to(device) FAILED:\n{traceback.format_exc()}", flush=True)
        raise

    try:
        t0 = time.perf_counter()
        print(f"[_transcribe] calling model.generate ...", flush=True)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=1024)
        new_tokens = outputs.shape[1] - input_len
        print(f"[_transcribe] model.generate OK {(time.perf_counter()-t0)*1000:.0f}ms new_tokens={new_tokens}", flush=True)
    except Exception:
        print(f"[_transcribe] model.generate FAILED:\n{traceback.format_exc()}", flush=True)
        raise

    try:
        text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        print(f"[_transcribe] decode OK text={repr(text[:120])}", flush=True)
        return text
    except Exception:
        print(f"[_transcribe] decode FAILED:\n{traceback.format_exc()}", flush=True)
        raise


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

    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
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

    text = _transcribe(audio_array)
    total_ms = (time.perf_counter() - req_start) * 1000
    print(f"[voxtral] {req_id} done total={total_ms:.0f}ms text_len={len(text)}")
    return {"text": text, "words": [], "languageCode": None}


@app.post("/transcribe-diarize")
async def transcribe_diarize(
    audio: UploadFile = File(...),
):
    """
    Upload audio → transcription + VAD sentence segmentation + per-segment emotion analysis.
    Returns structured segments: [{id, speaker, start, end, text, emotion, valence, arousal}]
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

    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)

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
    full_text = _transcribe(audio_array)
    print(f"[voxtral] {req_id} transcription done in {(time.perf_counter()-t0)*1000:.0f}ms text_len={len(full_text)}")

    # ── Step 2: VAD sentence segmentation ───────────────────────────────────
    t0 = time.perf_counter()
    raw_segs, seg_method = _segments_from_vad(audio_array, target_sr)
    print(f"[voxtral] {req_id} segmentation done in {(time.perf_counter()-t0)*1000:.0f}ms segs={len(raw_segs)}")

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
        "diarization_method": seg_method,
    }
