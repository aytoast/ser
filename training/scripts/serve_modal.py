"""Evoxtral API — Serverless inference on Modal.

Swagger UI: https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run/docs

Usage:
    # Deploy:
    modal deploy training/scripts/serve_modal.py

    # Test locally:
    modal serve training/scripts/serve_modal.py

    # Call the API (JSON response):
    curl -X POST https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run/transcribe \
        -F "file=@audio.wav"

    # Call the streaming API (Server-Sent Events):
    curl -N -X POST https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run/transcribe/stream \
        -F "file=@audio.wav"
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
        "transformers==4.56.0",
        "peft>=0.13.0",
        "accelerate>=1.0.0",
        "mistral-common",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "huggingface_hub",
        "safetensors",
        "sentencepiece",
        "fastapi",
        "python-multipart",
        "sse-starlette",
        gpu="A10G",
    )
    .env({"HF_HUB_CACHE": "/cache/huggingface"})
)

app = modal.App("evoxtral-api", image=image)
hf_cache = modal.Volume.from_name("evoxtral-hf-cache", create_if_missing=True)

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
ADAPTER_ID = "YongkangZOU/evoxtral-rl"


def _decode_audio(audio_bytes):
    """Decode audio bytes to float32 numpy array at 16kHz."""
    import numpy as np
    import soundfile as sf
    import librosa
    import io

    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    audio_array = audio_array.astype(np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    return audio_array


def _prepare_inputs(processor, audio_array, language, device):
    """Prepare model inputs from audio array."""
    import torch

    inputs = processor.apply_transcription_request(
        language=language,
        audio=[audio_array],
        format=["WAV"],
        model_id=MODEL_ID,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device, dtype=torch.bfloat16)
        if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
        else v.to(device)
        for k, v in inputs.items()
    }
    return inputs


@app.cls(
    gpu="A10G",
    volumes={"/cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    memory=65536,
    timeout=600,
)
class EvoxtralModel:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        print("Loading model...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        base_model = VoxtralForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        self.model.eval()
        print(f"Model loaded on {self.model.device}")

    @modal.asgi_app()
    def web(self):
        import torch
        import json
        import asyncio
        import numpy as np
        from threading import Thread
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
        from transformers import TextIteratorStreamer

        web_app = FastAPI(
            title="Evoxtral API",
            description=(
                "Expressive tagged transcription powered by Voxtral-Mini-3B + LoRA. "
                "Upload audio and get transcriptions with inline expressive tags like "
                "[sighs], [laughs], [whispers], etc.\n\n"
                "**Endpoints:**\n"
                "- `POST /transcribe` — Returns full transcription as JSON\n"
                "- `POST /transcribe/stream` — Streams tokens via Server-Sent Events (SSE)"
            ),
            version="2.0.0",
        )

        @web_app.get("/health", summary="Health check")
        async def health():
            return {"status": "ok", "model": "evoxtral-rl", "base": MODEL_ID}

        @web_app.post(
            "/transcribe",
            summary="Transcribe audio with expressive tags",
            response_description="JSON with transcription text",
        )
        async def transcribe(
            file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, etc.)"),
            language: str = Form("en", description="Language code (e.g. 'en', 'fr', 'es')"),
        ):
            audio_bytes = await file.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Empty audio file")

            try:
                audio_array = _decode_audio(audio_bytes)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}")

            inputs = _prepare_inputs(self.processor, audio_array, language, self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            transcription = self.processor.tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )

            return {
                "transcription": transcription,
                "language": language,
                "model": "evoxtral-rl",
            }

        @web_app.post(
            "/transcribe/stream",
            summary="Transcribe audio with streaming (SSE)",
            response_description="Server-Sent Events stream of transcription tokens",
        )
        async def transcribe_stream(
            file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, etc.)"),
            language: str = Form("en", description="Language code (e.g. 'en', 'fr', 'es')"),
        ):
            audio_bytes = await file.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Empty audio file")

            try:
                audio_array = _decode_audio(audio_bytes)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}")

            inputs = _prepare_inputs(self.processor, audio_array, language, self.model.device)

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generate_kwargs = dict(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                streamer=streamer,
            )

            thread = Thread(target=lambda: self.model.generate(**generate_kwargs))
            thread.start()

            async def event_generator():
                full_text = ""
                for token_text in streamer:
                    if token_text:
                        full_text += token_text
                        yield f"data: {json.dumps({'token': token_text})}\n\n"
                yield f"data: {json.dumps({'done': True, 'transcription': full_text, 'language': language, 'model': 'evoxtral-rl'})}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return web_app
