"""Evoxtral — Expressive Tagged Transcription Demo (ZeroGPU)."""

import torch
import spaces
import gradio as gr
import numpy as np
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
ADAPTER_ID = "YongkangZOU/evoxtral-rl"

# Load model on CPU at startup, ZeroGPU moves to GPU on demand
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = VoxtralForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()
print("Model loaded!")


@spaces.GPU
def transcribe(audio_input):
    """Transcribe audio with expressive tags."""
    if audio_input is None:
        return "Please upload or record an audio file."

    sr, audio_array = audio_input
    # Convert to float32 and mono if needed
    audio_array = audio_array.astype(np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    # Normalize to [-1, 1]
    if audio_array.max() > 1.0:
        audio_array = audio_array / 32768.0

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    inputs = processor.apply_transcription_request(
        language="en",
        audio=[audio_array],
        format=["WAV"],
        model_id=MODEL_ID,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device, dtype=torch.bfloat16)
        if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
        else v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    transcription = processor.tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )
    return transcription


EXAMPLES_TEXT = """
## What is Evoxtral?

Evoxtral is a fine-tuned version of [Voxtral-Mini-3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
that produces transcriptions enriched with **inline expressive audio tags** from the
[ElevenLabs v3 tag set](https://elevenlabs.io/docs/api-reference/text-to-speech).

### Standard ASR output:
> So I was thinking maybe we could try that new restaurant downtown.

### Evoxtral output:
> [nervous] So... I was thinking maybe we could [clears throat] try that new restaurant downtown? [laughs nervously]

### Supported tags include:
`[laughs]` `[sighs]` `[gasps]` `[whispers]` `[clears throat]` `[pause]` `[nervous]` `[frustrated]` `[excited]` `[sad]` `[calm]` `[stammers]` `[yawns]` and more.

### Results

| Metric | Base Voxtral | Evoxtral | Improvement |
|--------|-------------|----------|-------------|
| WER | 6.64% | **4.47%** | 32.7% better |
| Tag F1 | 22.0% | **67.2%** | 3x better |
"""

with gr.Blocks(title="Evoxtral — Expressive Tagged Transcription", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Evoxtral — Expressive Tagged Transcription")
    gr.Markdown("Upload or record audio to get a transcription with inline expressive tags like `[sighs]`, `[laughs]`, `[whispers]`, etc.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Audio Input",
                type="numpy",
                sources=["upload", "microphone"],
            )
            submit_btn = gr.Button("Transcribe", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Expressive Transcription",
                lines=8,
                show_copy_button=True,
            )

    submit_btn.click(fn=transcribe, inputs=audio_input, outputs=output_text)

    gr.Markdown(EXAMPLES_TEXT)

    gr.Markdown("""
---
Built for the **Mistral AI Online Hackathon 2026** (W&B Fine-Tuning Track) |
[Model](https://huggingface.co/YongkangZOU/evoxtral-lora) |
[W&B Dashboard](https://wandb.ai/yongkang-zou-ai/evoxtral) |
By Yongkang Zou
""")

demo.launch()
