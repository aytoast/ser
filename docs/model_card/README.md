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
  - rl
  - raft
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

**Two model variants available:**
- **[Evoxtral SFT](https://huggingface.co/YongkangZOU/evoxtral-lora)** — Best overall transcription accuracy (lowest WER)
- **[Evoxtral RL](https://huggingface.co/YongkangZOU/evoxtral-rl)** — Best expressive tag accuracy (highest Tag F1)

## What It Does

Standard ASR:
> So I was thinking maybe we could try that new restaurant downtown. I mean if you're free this weekend.

Evoxtral:
> [nervous] So... [stammers] I was thinking maybe we could... [clears throat] try that new restaurant downtown? [laughs nervously] I mean, if you're free this weekend?

## Training Pipeline

```
Base Voxtral-Mini-3B → SFT (LoRA, 3 epochs) → RL (RAFT, 1 epoch)
```

1. **SFT**: LoRA finetuning on 808 synthetic audio samples with expressive tags (lr=2e-4, 3 epochs)
2. **RL (RAFT)**: Rejection sampling — generate 4 completions per sample, score with rule-based reward (WER accuracy + Tag F1 - hallucination penalty), keep best, then SFT on curated data (lr=5e-5, 1 epoch)

This follows the approach from [GRPO for Speech Recognition](https://arxiv.org/abs/2509.01939) and Voxtral's own SFT→DPO training recipe.

## Evaluation Results

Evaluated on 50 held-out test samples. Full benchmark (Evoxtral-Bench) with 7 metrics:

### Core Metrics — Base vs SFT vs RL

| Metric | Base Voxtral | Evoxtral SFT | Evoxtral RL | Best |
|--------|-------------|-------------|------------|------|
| **WER** | 6.64% | **4.47%** | 5.12% | SFT |
| **CER** | 2.72% | **1.23%** | 1.48% | SFT |
| **Tag F1** | 22.0% | 67.2% | **69.4%** | RL |
| **Tag Precision** | 22.0% | 67.4% | **68.5%** | RL |
| **Tag Recall** | 22.0% | 69.4% | **72.7%** | RL |
| **Emphasis F1** | 42.0% | 84.0% | **86.0%** | RL |
| **Tag Hallucination** | 0.0% | **19.3%** | 20.2% | SFT |

**SFT** excels at raw transcription accuracy (best WER/CER). **RL** further improves expressive tag generation (+2.2% Tag F1, +3.3% Tag Recall, +2% Emphasis F1) at a small cost to WER.

### Per-Tag F1 Breakdown (SFT → RL)

| Tag | SFT F1 | RL F1 | Change | Support |
|-----|--------|-------|--------|---------|
| `[sighs]` | 1.000 | **1.000** | — | 9 |
| `[clears throat]` | 0.889 | **1.000** | +12.5% | 8 |
| `[gasps]` | 0.957 | **0.957** | — | 12 |
| `[pause]` | 0.885 | **0.902** | +1.9% | 25 |
| `[nervous]` | 0.800 | **0.846** | +5.8% | 13 |
| `[stammers]` | 0.889 | 0.842 | -5.3% | 8 |
| `[laughs]` | 0.800 | **0.815** | +1.9% | 12 |
| `[sad]` | 0.667 | **0.750** | +12.4% | 4 |
| `[whispers]` | 0.636 | **0.667** | +4.9% | 13 |
| `[crying]` | 0.750 | 0.571 | -23.9% | 5 |
| `[excited]` | 0.615 | 0.571 | -7.2% | 5 |
| `[shouts]` | 0.400 | **0.500** | +25.0% | 3 |
| `[calm]` | 0.200 | **0.400** | +100% | 6 |
| `[frustrated]` | 0.444 | 0.444 | — | 3 |
| `[angry]` | 0.667 | 0.667 | — | 2 |
| `[confused]` | 0.000 | 0.000 | — | 1 |
| `[scared]` | 0.000 | 0.000 | — | 1 |

RL improved 9 tags, kept 4 stable, and regressed 3. Biggest gains on [clears throat] (+12.5%), [calm] (+100%), [sad] (+12.4%), and [shouts] (+25%).

## Training Details

### SFT Stage

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

### RL Stage (RAFT)

| Parameter | Value |
|-----------|-------|
| Method | Rejection sampling + SFT (RAFT) |
| Samples per input | 4 (temperature=0.7, top_p=0.9) |
| Reward function | 0.4×(1-WER) + 0.4×Tag_F1 + 0.2×(1-hallucination) |
| Curated samples | 727 (bottom 10% filtered, reward > 0.954) |
| Avg reward | 0.980 |
| Learning rate | 5e-5 |
| Epochs | 1 |
| Final loss | 0.021 |
| Training time | ~7 minutes |

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
# Use "YongkangZOU/evoxtral-lora" for SFT or "YongkangZOU/evoxtral-rl" for RL
adapter_id = "YongkangZOU/evoxtral-rl"

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
- [SFT Training](https://wandb.ai/yongkang-zou-ai/evoxtral/runs/t8ak7a20)
- [RL Training (RAFT)](https://wandb.ai/yongkang-zou-ai/evoxtral)
- [Base model eval](https://wandb.ai/yongkang-zou-ai/evoxtral/runs/bvqa4ioo)
- [SFT model eval](https://wandb.ai/yongkang-zou-ai/evoxtral/runs/ayx4ldyd)
- [RL model eval](https://wandb.ai/yongkang-zou-ai/evoxtral)
- [Project dashboard](https://wandb.ai/yongkang-zou-ai/evoxtral)

## Supported Tags

The model can produce any tag from the ElevenLabs v3 expressive tag set, including:

`[laughs]` `[sighs]` `[gasps]` `[clears throat]` `[whispers]` `[sniffs]` `[pause]` `[nervous]` `[frustrated]` `[excited]` `[sad]` `[angry]` `[calm]` `[stammers]` `[yawns]` and more.

## Limitations

- Trained on synthetic (TTS-generated) audio, not natural speech recordings
- ~20% tag hallucination rate — model occasionally predicts tags not in the reference
- Rare/subtle tags ([calm], [confused], [scared]) have low accuracy due to limited training examples
- RL variant trades ~0.65% WER for better tag accuracy
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
