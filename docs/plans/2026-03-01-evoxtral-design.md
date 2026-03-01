# Evoxtral Design Document

## Overview

**Evoxtral** (Emotion + Voxtral) finetunes Mistral's Voxtral-Mini-3B to produce transcriptions with inline ElevenLabs v3 audio tags. The model takes speech audio and outputs tagged text like:

```
[excited] I can't BELIEVE this happened! [laughs] It was absolutely incredible...
```

This tagged transcription can be directly fed into ElevenLabs v3 TTS to reproduce emotional delivery -- creating a speech-to-expressive-speech pipeline.

## Problem

Transcription models strip all emotion, prosody, and non-verbal sounds. ElevenLabs v3 has powerful audio tags (`[laughs]`, `[whispers]`, `[excited]`, etc.) but no model automatically generates them from audio. Evoxtral bridges this gap.

## Architecture

### Three Pillars

```
[Audio Input] -> [Voxtral-Mini-3B + LoRA] -> [Tagged Transcription]
                                                      |
                                                      v
                                           [ElevenLabs v3 TTS]
                                                      |
                                                      v
                                           [Expressive Audio Output]
```

### Pillar 1: Synthetic Training Data Pipeline

Generate training data by reversing the pipeline:

1. Curate diverse tagged scripts (LLM-assisted)
2. Synthesize audio via ElevenLabs v3 TTS (multiple API keys for rate limiting)
3. Produce `(audio, tagged_text)` training pairs
4. Target: 1,000 pairs

The tagged text serves as both input to ElevenLabs (for synthesis) and ground truth labels (for training).

#### Dataset Composition (1,000 examples)

| Slice | % | Count | Tag Density | Purpose |
|-------|---|-------|-------------|---------|
| Plain ASR | 25% | 250 | Zero tags | Prevent tag hallucination, preserve base WER |
| Light Tags | 25% | 250 | 1-2 tags | Learn subtle, sparse tagging |
| Moderate Tags | 25% | 250 | 3-4 tags | Core expressive transcription |
| Dense Tags | 15% | 150 | 5+ tags | Highly emotional/dramatic speech |
| Edge Cases | 10% | 100 | Mixed | Ambiguous emotion, back-to-back tags, boundary tags |

Rationale: 10% negative examples is optimal to prevent hallucination (hallucination tax research). A ~1:3 ratio of plain:tagged data preserves base WER with <1pp degradation (Apple/data mixing research). Tag density gradient prevents over/under-generation (SSML research).

#### Tag Frequency Targets (across 800 tagged examples)

Rare tags (`[gasps]`, `[stammers]`, `[crying]`) oversampled 3-5x. Common tags (`[pause]`, CAPS) undersampled to ~0.5x natural frequency. Every tag type appears in at least 3% of total examples.

#### Quality Gates

- ROUGE-L < 0.7 between any two scripts (enforce diversity)
- 6-8 ElevenLabs voices, round-robin, no voice >20% of samples
- No single domain (conversation, monologue, etc.) exceeds 25%
- Audio validation: discard garbled/failed TTS outputs
- Dataset fields: `{audio, tagged_text, plain_text, tag_list, slice_type}`

### Pillar 2: Two-Stage Finetuning

#### Stage 1: SFT + LoRA + NEFTune (primary)

- **Base model**: `mistralai/Voxtral-Mini-3B-2507` (3B params, ~9.5GB VRAM, Apache 2.0)
- **Method**: LoRA via HuggingFace `peft`
- **LoRA config**: rank=64, alpha=128, dropout=0.05
- **Target modules**: LLM linear layers + `multi_modal_projector.linear_1/linear_2`
- **NEFTune**: noise_alpha=5 (noisy embeddings, +35% quality for free)
- **Label masking**: Loss computed only on tagged transcription output tokens
- **Epochs**: 1 (multi-epoch degrades instruction tuning per Raschka's research)
- **Tracking**: Weights & Biases (`report_to="wandb"`)
- **Hosting**: Push LoRA adapter to HuggingFace Hub, serve via HF Inference Endpoints

#### Stage 2: ORPO or SimPO (conditional, if tag F1 < 0.6)

- Generate 200-500 preference pairs from Stage 1 model outputs
- ORPO: single-stage, reference-model-free, combines SFT + preference
- SimPO: reference-free, simpler than DPO, uses avg log prob as implicit reward
- Quick 1-epoch pass to refine tag placement and reduce hallucination

#### Why Voxtral-Mini-3B

| Model | Params | VRAM | Finetuning Support | Verdict |
|-------|--------|------|--------------------|---------|
| Voxtral Mini 3B | 3B | ~9.5GB | Proven LoRA scripts | Best choice |
| Voxtral Small 24B | 24B | ~55GB | Feasible but slow | Too heavy for 24h |
| Voxtral Mini 4B Realtime | 4B | ~16GB | vLLM only, no precedent | Too risky |

### Pillar 3: Evoxtral-Bench (Custom Evaluation)

No existing benchmark covers tagged transcription. Evoxtral-Bench has 3 layers:

**Layer 1 -- Text Accuracy (WER)**
- Strip all `[tags]` from output
- Compute WER via `jiwer` against plain-text ground truth
- Target: < 15% WER on clean English audio

**Layer 2 -- Tag Accuracy (Classification Metrics)**
- Tag Presence F1: correct set of tags per segment
- Tag Type Accuracy: semantic grouping (e.g., `[laughs]` ~ `[giggles]`)
- Tag Position Score: tag placed within N words of ground-truth position

**Layer 3 -- Round-Trip Audio Quality**
- Feed tagged output into ElevenLabs v3 TTS
- Compare synthesized audio to original using perceptual metrics
- Validates that tags actually improve expressiveness

Ground truth comes from the synthetic dataset -- we know exactly what tags produced each audio sample.

## Target Audio Tags

Focused set of ElevenLabs v3 tags:

| Category | Tags |
|----------|------|
| Emotions | `[excited]`, `[sad]`, `[angry]`, `[nervous]`, `[calm]`, `[frustrated]` |
| Non-verbal | `[laughs]`, `[sighs]`, `[gasps]`, `[clears throat]`, `[crying]` |
| Delivery | `[whispers]`, `[shouts]`, `[stammers]` |
| Pauses | `[pause]`, ellipses `...` |
| Emphasis | CAPITALIZATION of stressed words |

## System Components

### Backend (FastAPI + Python)

- `POST /transcribe` -- upload audio, get tagged transcription
- `WS /transcribe/stream` -- streaming tagged transcription
- Model loaded via `transformers` + `peft` (LoRA adapter from HF Hub)
- Speaker diarization via `pyannote.audio`

### Frontend (Next.js + shadcn)

- Record/upload audio
- Display tagged transcription with visual tag highlights
- Playback re-synthesized audio via ElevenLabs
- Side-by-side comparison: original vs re-synthesized

### Data Pipeline

- Script generator (LLM creates diverse tagged dialogues)
- ElevenLabs batch synthesizer (multi-key rotation for rate limits)
- Dataset formatter (HuggingFace datasets format)

### Eval Suite (Evoxtral-Bench)

- `jiwer` for WER computation
- Custom tag parser and F1 scorer
- ElevenLabs round-trip comparator
- W&B integration for logging eval results

## W&B Integration

- `Trainer(report_to="wandb")` for training metrics
- `WANDB_LOG_MODEL="checkpoint"` for model versioning
- `wandb.Audio()` for sample predictions during eval
- Optional: Bayesian sweep for learning rate / LoRA rank

## HuggingFace Hosting

- Push LoRA adapter to HF Hub under `mistral-hackaton-2026` org
- Serve via HF Inference Endpoints ($20 HF credits available)
- Model card with eval results and usage examples

## 24h Timeline

| Phase | Hours | What | Parallel? |
|-------|-------|------|-----------|
| Data Generation | 0-4h | Tagged scripts + ElevenLabs synthesis | -- |
| Training | 4-10h | LoRA finetune, W&B tracking | Yes with frontend |
| Eval | 10-12h | Run Evoxtral-Bench, iterate | -- |
| Backend | 8-14h | FastAPI, model serving from HF | Yes with frontend |
| Frontend | 10-18h | Next.js UI | Yes with backend |
| Integration | 18-22h | End-to-end testing, polish | -- |
| Demo | 22-24h | Record demo, model card, submission | -- |

## Tech Stack

- Python 3.11+, FastAPI, WebSockets
- `transformers` + `peft` + `torchaudio`
- `wandb` for experiment tracking
- `jiwer` for WER evaluation
- `pyannote.audio` for diarization
- ElevenLabs Python SDK for TTS
- Next.js, shadcn/ui, Framer Motion
- HuggingFace Hub for model hosting

## References

- [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
- [Finetune-Voxtral-ASR](https://github.com/Deep-unlearning/Finetune-Voxtral-ASR)
- [ElevenLabs v3 Audio Tags](https://elevenlabs.io/blog/v3-audiotags)
- [ElevenLabs API Docs](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
- [W&B HuggingFace Integration](https://docs.wandb.ai/models/integrations/huggingface)
