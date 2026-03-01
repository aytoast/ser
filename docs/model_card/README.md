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

Evaluated on 50 held-out test samples. Full benchmark (Evoxtral-Bench) with 7 metrics:

### Core Metrics

| Metric | Base Voxtral | Evoxtral (finetuned) | Improvement |
|--------|-------------|---------------------|-------------|
| **WER** (Word Error Rate) | 6.64% | **4.47%** | 32.7% better |
| **CER** (Character Error Rate) | 2.72% | **1.23%** | 54.8% better |
| **Tag F1** | 22.0% | **67.2%** | 3.1x better |
| **Tag Precision** | 22.0% | **67.4%** | 3.1x better |
| **Tag Recall** | 22.0% | **69.4%** | 3.2x better |
| **Emphasis F1** (CAPS words) | 42.0% | **84.0%** | 2.0x better |
| **Tag Hallucination Rate** | 0.0% | 19.3% | trade-off |

The finetuned model dramatically improves expressive tag generation and emphasis detection while also improving raw transcription accuracy. The 19.3% hallucination rate indicates the model occasionally predicts tags not present in the reference — a known trade-off when optimizing for tag recall.

### Per-Tag F1 Breakdown

| Tag | F1 | Precision | Recall | Support |
|-----|----|-----------|--------|---------|
| `[sighs]` | **1.000** | 1.000 | 1.000 | 9 |
| `[gasps]` | **0.957** | 1.000 | 0.917 | 12 |
| `[clears throat]` | 0.889 | 0.800 | 1.000 | 8 |
| `[stammers]` | 0.889 | 0.800 | 1.000 | 8 |
| `[pause]` | 0.885 | 0.852 | 0.920 | 25 |
| `[laughs]` | 0.800 | 0.769 | 0.833 | 12 |
| `[nervous]` | 0.800 | 0.833 | 0.769 | 13 |
| `[crying]` | 0.750 | 1.000 | 0.600 | 5 |
| `[sad]` | 0.667 | 0.600 | 0.750 | 4 |
| `[angry]` | 0.667 | 1.000 | 0.500 | 2 |
| `[whispers]` | 0.636 | 0.778 | 0.538 | 13 |
| `[excited]` | 0.615 | 0.500 | 0.800 | 5 |
| `[frustrated]` | 0.444 | 0.333 | 0.667 | 3 |
| `[shouts]` | 0.400 | 0.500 | 0.333 | 3 |
| `[calm]` | 0.200 | 0.250 | 0.167 | 6 |
| `[confused]` | 0.000 | 0.000 | 0.000 | 1 |
| `[scared]` | 0.000 | 0.000 | 0.000 | 1 |

**Best performing**: Tags with clear acoustic signals ([sighs], [gasps], [clears throat]) achieve near-perfect F1. **Weakest**: Subtle emotional states ([calm], [confused], [scared]) with low training support.

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

## API

A serverless API with Swagger UI is available on Modal:

```bash
curl -X POST https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run/transcribe \
    -F "file=@audio.wav"
```

- [Swagger UI](https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run/docs)
- [Live Demo (HF Space)](https://huggingface.co/spaces/YongkangZOU/evoxtral)

## W&B Tracking

All training and evaluation runs are tracked on Weights & Biases:
- [Training run](https://wandb.ai/yongkang-zou-ai/evoxtral/runs/t8ak7a20)
- [Base model eval](https://wandb.ai/yongkang-zou-ai/evoxtral/runs/bvqa4ioo)
- [Finetuned model eval](https://wandb.ai/yongkang-zou-ai/evoxtral/runs/ayx4ldyd)
- [Project dashboard](https://wandb.ai/yongkang-zou-ai/evoxtral)

## Supported Tags

The model can produce any tag from the ElevenLabs v3 expressive tag set, including:

`[laughs]` `[sighs]` `[gasps]` `[clears throat]` `[whispers]` `[sniffs]` `[pause]` `[nervous]` `[frustrated]` `[excited]` `[sad]` `[angry]` `[calm]` `[stammers]` `[yawns]` and more.

## Limitations

- Trained on synthetic (TTS-generated) audio, not natural speech recordings
- 19.3% tag hallucination rate — model occasionally predicts tags not in the reference
- Rare/subtle tags ([calm], [confused], [scared]) have low accuracy due to limited training examples
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
