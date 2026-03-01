---
library_name: peft
base_model: mistralai/Voxtral-Mini-3B-2507
tags:
  - voxtral
  - lora
  - speech-recognition
  - expressive-transcription
  - audio
  - mistral
  - hackathon
datasets:
  - custom
language:
  - en
license: apache-2.0
pipeline_tag: automatic-speech-recognition
---

# Evoxtral LoRA — Expressive Tagged Transcription

A LoRA adapter for [Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) that produces transcriptions enriched with inline expressive audio tags from the [ElevenLabs v3 tag set](https://elevenlabs.io/docs/api-reference/text-to-speech).

Built for the **Mistral AI Online Hackathon 2026** (W&B Fine-Tuning Track).

## What It Does

Standard ASR:
> So I was thinking maybe we could try that new restaurant downtown. I mean if you're free this weekend.

Evoxtral:
> [nervous] So... [stammers] I was thinking maybe we could... [clears throat] try that new restaurant downtown? [laughs nervously] I mean, if you're free this weekend?

## Evaluation Results

| Metric | Base Voxtral | Evoxtral (finetuned) | Improvement |
|--------|-------------|---------------------|-------------|
| **WER** (Word Error Rate) | 6.64% | **4.47%** | 32.7% better |
| **Tag F1** (Expressive Tag Accuracy) | 22.0% | **67.2%** | 3x better |

Evaluated on 50 held-out test samples. The finetuned model dramatically improves expressive tag generation while also improving raw transcription accuracy.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `mistralai/Voxtral-Mini-3B-2507` |
| Method | LoRA (PEFT) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o_proj, gate/up/down_proj, multi_modal_projector |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Epochs | 3 |
| Batch size | 2 (effective 16 with grad accum 8) |
| NEFTune noise alpha | 5.0 |
| Precision | bf16 |
| GPU | NVIDIA A10G (24GB) |
| Training time | ~25 minutes |
| Trainable params | 124.8M / 4.8B (2.6%) |

## Dataset

Custom synthetic dataset of 1,010 audio samples generated with ElevenLabs TTS v3:
- **808** train / **101** validation / **101** test
- Each sample has audio + tagged transcription with inline ElevenLabs v3 expressive tags
- Tags include: `[sighs]`, `[laughs]`, `[whispers]`, `[nervous]`, `[frustrated]`, `[clears throat]`, `[pause]`, `[excited]`, and more
- Audio encoder (Whisper-based) was frozen during training

## Usage

```python
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from peft import PeftModel

repo_id = "mistralai/Voxtral-Mini-3B-2507"
adapter_id = "YongkangZOU/evoxtral-lora"

processor = AutoProcessor.from_pretrained(repo_id)
base_model = VoxtralForConditionalGeneration.from_pretrained(
    repo_id, dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_id)

# Transcribe audio with expressive tags
inputs = processor.apply_transcription_request(
    language="en",
    audio=["path/to/audio.wav"],
    format=["WAV"],
    model_id=repo_id,
    return_tensors="pt",
)
inputs = inputs.to(model.device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
transcription = processor.batch_decode(
    outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0]
print(transcription)
# [nervous] So... I was thinking maybe we could [clears throat] try that new restaurant downtown?
```

## W&B Tracking

All training and evaluation runs are tracked on Weights & Biases:
- [Training run](https://wandb.ai/yongkang-zou-ai/evoxtral/runs/t8ak7a20)
- [Base model eval](https://wandb.ai/yongkang-zou-ai/evoxtral/runs/f9l2zwvs)
- [Finetuned model eval](https://wandb.ai/yongkang-zou-ai/evoxtral/runs/b32c74im)
- [Project dashboard](https://wandb.ai/yongkang-zou-ai/evoxtral)

## Supported Tags

The model can produce any tag from the ElevenLabs v3 expressive tag set, including:

`[laughs]` `[sighs]` `[gasps]` `[clears throat]` `[whispers]` `[sniffs]` `[pause]` `[nervous]` `[frustrated]` `[excited]` `[sad]` `[angry]` `[calm]` `[stammers]` `[yawns]` and more.

## Limitations

- Trained on synthetic (TTS-generated) audio, not natural speech recordings
- Tag F1 of 67.2% means ~1/3 of tags may be missed or misplaced
- English only
- Best results on conversational and emotionally expressive speech

## Citation

```bibtex
@misc{evoxtral2026,
  title={Evoxtral: Expressive Tagged Transcription with Voxtral},
  author={Yongkang Zou},
  year={2026},
  url={https://huggingface.co/YongkangZOU/evoxtral-lora}
}
```
