# Evoxtral — Research References & Sources

References for the technical report covering expressive tagged transcription, ASR evaluation, and post-training methods.

## Core Model

- **Voxtral: An Audio-Language Model** — Mistral AI, 2025. Voxtral Mini and Small are open-weights audio chat models trained with SFT + DPO + Online DPO. Online DPO delivers crisper grounding, fewer hallucinations, and more helpful responses.
  - Paper: https://arxiv.org/abs/2507.13264
  - Model: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507

## RL / Post-Training for ASR

- **Group Relative Policy Optimization for Speech Recognition** — Proposes GRPO with rule-based rewards for LLM-based ASR. Achieved 18% relative WER reduction on AMI-IHM and 27.9% on AMI-SDM compared to SFT-adapted models.
  - Paper: https://arxiv.org/abs/2509.01939

- **Advancing Speech Understanding in Speech-Aware Language Models with GRPO** — Applies GRPO to large audio language models, investigating different rule-based reward functions and RL data construction strategies.
  - Paper: https://arxiv.org/abs/2509.16990

- **Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering** — Demonstrates RL (GRPO) achieves state-of-the-art on audio QA tasks, outperforming SFT baselines.
  - Paper: https://arxiv.org/abs/2503.11197

- **Explore the Reinforcement Learning for LLM-based ASR and TTS System** — Surveys RL applications to speech models, noting that while RL has enhanced text-based LLMs, application to ASR/TTS remains underexplored.
  - Paper: https://arxiv.org/abs/2509.18569

## ASR Evaluation Metrics

- **Speech Recognition Accuracy: Production Metrics & Optimization 2025** — Deepgram. Covers WER, CER, Keyword Recall Rate (KRR), Real-Time Factor (RTF), and end-to-end latency. Production systems need blended metrics depending on use case.
  - Source: https://deepgram.com/learn/speech-recognition-accuracy-production-metrics

- **Moving Beyond Word Error Rate to Evaluate ASR in Clinical Samples** — Argues WER alone is insufficient; error type, meaning, and context matter. A single substitution can drastically change intent with the same WER penalty.
  - Paper: https://www.sciencedirect.com/science/article/pii/S0165178125003385

- **Measuring the Accuracy of Automatic Speech Recognition Solutions** — ACM survey on ASR evaluation methodology, limitations of WER/CER, and alternative metrics.
  - Paper: https://dl.acm.org/doi/10.1145/3636513

- **On the Robust Approximation of ASR Metrics** — ACL 2025. Novel label-free approach for approximating ASR performance using multimodal embeddings.
  - Paper: https://arxiv.org/abs/2502.12408

- **ProfASR-Bench: A Professional-talk ASR Dataset for High-Stakes Applications** — Evaluation suite supporting conventional metrics plus entity-aware scores and slice-wise reporting by accent and gender.
  - Paper: https://arxiv.org/abs/2512.23686

## Expressive Speech & TTS

- **ElevenLabs v3 Text-to-Speech** — TTS model supporting inline expressive audio tags for emotions, non-verbal sounds, and delivery cues. Tag set used as target vocabulary for Evoxtral.
  - Docs: https://elevenlabs.io/docs/api-reference/text-to-speech

## Fine-Tuning Frameworks

- **ms-swift** — ModelScope framework supporting SFT/DPO/GRPO for 600+ LLMs and 300+ MLLMs. AAAI 2025.
  - GitHub: https://github.com/modelscope/ms-swift

- **OpenRLHF** — Scalable agentic RL framework based on Ray, supporting PPO, DAPO, REINFORCE++, and more.
  - GitHub: https://github.com/OpenRLHF/OpenRLHF

- **PEFT (Parameter-Efficient Fine-Tuning)** — HuggingFace library for LoRA and other adapter methods.
  - GitHub: https://github.com/huggingface/peft

## Evaluation Methodology

- **jiwer** — Python library for WER, CER, and other ASR metrics based on edit distance.
  - GitHub: https://github.com/jitsi/jiwer

## Our Benchmark: Evoxtral-Bench

Metrics computed by `scripts/train_modal.py::evaluate()`:

| Metric | Description | Direction |
|--------|-------------|-----------|
| **WER** | Word Error Rate on plain text (tags stripped) | lower = better |
| **CER** | Character Error Rate on plain text | lower = better |
| **Tag F1** | F1 score for tag extraction (multiset intersection) | higher = better |
| **Tag Precision** | Fraction of predicted tags that match reference | higher = better |
| **Tag Recall** | Fraction of reference tags captured by prediction | higher = better |
| **Tag Hallucination Rate** | Fraction of predicted tags not in reference at all | lower = better |
| **Emphasis F1** | F1 for CAPITALIZED emphasis words | higher = better |
| **Per-Tag F1** | Breakdown by individual tag type (e.g. [laughs], [sighs]) | higher = better |

### Why These Metrics

- **WER + CER**: Standard ASR quality. CER is more granular and catches character-level errors WER misses.
- **Tag Precision vs Recall**: F1 alone hides whether the model hallucinates tags (low precision) or misses them (low recall). Judges care about this distinction.
- **Tag Hallucination Rate**: Critical for downstream TTS — hallucinated tags produce wrong prosody/effects.
- **Per-Tag Breakdown**: Shows which expressive cues the model handles well vs struggles with, informing future data collection.
- **Emphasis F1**: CAPS words (e.g., "BELIEVE", "NEVER") carry prosodic emphasis in ElevenLabs v3; measuring this separately tracks delivery accuracy.
