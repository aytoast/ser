"""Speech-to-expressive-speech agent pipeline with full W&B Weave tracing.

Pipeline:
    1. Audio input -> Voxtral transcription (with ElevenLabs v3 expressive tags)
    2. Tagged text -> ElevenLabs v3 TTS -> expressive audio output

Every step is traced with @weave.op() decorators for observability.
"""

import os
import re
import time
import tempfile
from pathlib import Path
from typing import Optional

import httpx
import torch
import weave
import librosa
from dotenv import load_dotenv
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from peft import PeftModel

load_dotenv()

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
ELEVENLABS_MODEL_ID = "eleven_v3"
TRANSCRIPTION_PROMPT = "Transcribe this audio with expressive tags."

VOICE_POOL = [
    {"id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Roger", "gender": "male", "age": "middle_aged"},
    {"id": "cjVigY5qzO86Huf0OWal", "name": "Eric", "gender": "male", "age": "middle_aged"},
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "gender": "female", "age": "young"},
    {"id": "XrExE9yKIg1WjnnlVkGX", "name": "Matilda", "gender": "female", "age": "middle_aged"},
    {"id": "TX3LPaxmHKxFdv7VOQHJ", "name": "Liam", "gender": "male", "age": "young"},
    {"id": "cgSgspJ2msm6clMCkdW9", "name": "Jessica", "gender": "female", "age": "young"},
    {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George", "gender": "male", "age": "middle_aged"},
    {"id": "pFZP5JQG7iQjIQuC4Bku", "name": "Lily", "gender": "female", "age": "middle_aged"},
]

TAG_PATTERN = re.compile(r"\[([^\]]+)\]")


def get_api_keys() -> list[str]:
    """Load all ElevenLabs API keys from environment."""
    keys = []
    for key_name, value in sorted(os.environ.items()):
        if key_name.startswith("ELEVENLABS_API_KEY"):
            keys.append(value)
    if not keys:
        raise ValueError("No ELEVENLABS_API_KEY* found in environment")
    return keys


def strip_tags(tagged_text: str) -> str:
    """Remove all bracket tags from text, returning plain transcription."""
    return TAG_PATTERN.sub("", tagged_text).strip()


def extract_tags(tagged_text: str) -> list[str]:
    """Extract all bracket tags from tagged text."""
    return TAG_PATTERN.findall(tagged_text)


# ---------------------------------------------------------------------------
# Initialize Weave project
# ---------------------------------------------------------------------------
weave.init("evoxtral")


class SpeechAgent(weave.Model):
    """End-to-end speech-to-expressive-speech agent.

    Transcribes audio with Voxtral (optionally with a LoRA adapter) to produce
    ElevenLabs v3 tagged text, then synthesizes expressive audio via ElevenLabs TTS.
    """

    adapter_path: Optional[str] = None
    default_voice_id: str = VOICE_POOL[0]["id"]

    # Private attributes (not serialized by weave.Model / Pydantic)
    _model: object = None
    _processor: object = None
    _device: object = None
    _api_keys: list = []

    def model_post_init(self, __context):
        """Load model, processor, and API keys after initialization."""
        self._load_model()
        self._api_keys = get_api_keys()

    def _load_model(self):
        """Load Voxtral model (with optional LoRA adapter)."""
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else "cpu"

        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        base_model = VoxtralForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map=device_map,
        )

        if self.adapter_path is not None:
            print(f"Loading LoRA adapter from {self.adapter_path}")
            self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            self._model = base_model

        self._model.eval()
        self._device = next(self._model.parameters()).device
        print(f"Model loaded on {self._device} (dtype={dtype})")

    @weave.op()
    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio to tagged text using Voxtral.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            dict with keys: tagged_text, plain_text, tags
        """
        # Load audio
        audio_array, sr = librosa.load(audio_path, sr=16000)

        # Build conversation for the processor
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": TRANSCRIPTION_PROMPT},
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move inputs to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
        else:
            inputs = inputs.to(self._device)

        # Generate
        with torch.no_grad():
            if isinstance(inputs, dict):
                output_ids = self._model.generate(**inputs, max_new_tokens=512)
            else:
                output_ids = self._model.generate(inputs, max_new_tokens=512)

        # Decode — skip the input tokens to get only the generated response
        if isinstance(inputs, dict):
            input_len = inputs["input_ids"].shape[-1]
        else:
            input_len = inputs.shape[-1]

        generated_ids = output_ids[:, input_len:]
        tagged_text = self._processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        plain_text = strip_tags(tagged_text)
        tags = extract_tags(tagged_text)

        return {
            "tagged_text": tagged_text,
            "plain_text": plain_text,
            "tags": tags,
        }

    @weave.op()
    def synthesize(self, tagged_text: str, voice_id: str | None = None) -> dict:
        """Synthesize tagged text to expressive audio via ElevenLabs v3 TTS.

        Args:
            tagged_text: Text with ElevenLabs v3 bracket tags.
            voice_id: ElevenLabs voice ID. Defaults to the agent's default voice.

        Returns:
            dict with keys: audio_path, duration_ms
        """
        if voice_id is None:
            voice_id = self.default_voice_id

        api_key = self._api_keys[0]

        start = time.time()

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": tagged_text,
                    "model_id": ELEVENLABS_MODEL_ID,
                    "output_format": "mp3_44100_128",
                },
            )

        elapsed_ms = (time.time() - start) * 1000

        if response.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs TTS failed (HTTP {response.status_code}): {response.text[:200]}"
            )

        # Write output audio to a temp file
        output_dir = Path(tempfile.gettempdir()) / "evoxtral_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"synth_{int(time.time() * 1000)}.mp3")

        with open(output_path, "wb") as f:
            f.write(response.content)

        return {
            "audio_path": output_path,
            "duration_ms": round(elapsed_ms, 2),
        }

    @weave.op()
    def predict(self, audio_path: str) -> dict:
        """Full pipeline: transcribe audio then synthesize expressive speech.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            dict with keys: transcription (dict), synthesis (dict)
        """
        transcription = self.transcribe(audio_path)
        synthesis = self.synthesize(transcription["tagged_text"])

        return {
            "transcription": transcription,
            "synthesis": synthesis,
        }


if __name__ == "__main__":
    import sys

    agent = SpeechAgent(adapter_path=sys.argv[1] if len(sys.argv) > 1 else None)
    result = agent.predict(sys.argv[2] if len(sys.argv) > 2 else "test.mp3")
    print(result)
