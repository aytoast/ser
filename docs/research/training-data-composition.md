# Training Data Composition & Balancing for Evoxtral Finetuning

Research summary on best practices for training data composition when finetuning LLMs,
applied to the Evoxtral use case (LoRA finetuning Voxtral-Mini-3B to produce tagged transcriptions).

---

## 1. Data Mixing Strategies for SFT/LoRA Finetuning

### Optimal Ratios of Tagged vs Plain Data

The single most important finding across the literature: **always include plain/untagged examples
in your training mix.** Training exclusively on tagged transcriptions will cause the model to
hallucinate tags everywhere and degrade base transcription quality.

**Concrete ratios from research:**

| Mix Ratio (Task:Original) | Source | Result |
|---------------------------|--------|--------|
| 1:1 (50% new, 50% original) | [Mixed Training for Math Reasoning](https://arxiv.org/html/2512.13706) | Best balance -- full new-task performance with only 0.7pp original-task degradation |
| 3:1 (75% new, 25% original) | Same study | New-task performance maintained, original task drops ~1.4pp |
| 7:1 (87.5% new, 12.5% original) | Same study | Still effective, original task drops ~2.5pp |
| 15:1 (93.8% new, 6.2% original) | Same study | Minimum viable -- original task drops ~3.2pp but still far better than 0% |

**For Evoxtral specifically:** With a target of 500-1000 tagged training pairs, aim for:
- **60-70% tagged transcriptions** (emotion tags, non-verbal markers, delivery cues)
- **30-40% plain transcriptions** (standard ASR output, no tags at all)

This ratio prevents the model from learning "always add tags" and preserves base transcription quality.

### Preventing Tag Hallucination

Research on preventing hallucination during finetuning is directly applicable to preventing
over-generation of audio tags.

**Key findings from [The Hallucination Tax of Reinforcement Finetuning](https://arxiv.org/html/2505.13988):**
- Standard finetuning can reduce refusal rates by >80%, meaning models become overconfident
- Tested mixing ratios of 0%, 1%, 10%, 30%, 50% "unanswerable" (negative) examples
- **10% negative examples was the optimal ratio** -- restored appropriate refusal behavior while maintaining task accuracy
- Higher ratios (30-50%) degraded performance on the primary task

**Applied to Evoxtral:** Include ~10-15% of training examples where the audio is emotionally
neutral/flat but the ground truth has NO tags (just plain text). This teaches the model that
not every utterance needs tags.

**Additional anti-hallucination strategies:**
- Train on "familiar, low-perplexity data" -- using high-perplexity examples increases hallucination ([Unfamiliar Finetuning Examples](https://arxiv.org/html/2403.05612v1))
- Include examples where the model must produce a balanced positive/negative ratio of tags ([Robust Instruction Tuning](https://arxiv.org/abs/2306.14565))
- Ensure tag density varies naturally across training examples (some heavily tagged, some sparse)

### The "Cocktail Effect" in Data Mixing

Research on [Data Mixing Optimization for SFT](https://arxiv.org/html/2508.11953v1) found a
"cocktail effect": diverse training data outperforms single-domain approaches. For domain-specific
models, including general instruction data alongside specialized content improved results. A medical
chatbot achieved best performance with **67.7% general data (Alpaca-GPT4) and 32.3% domain data
(PubMedQA).**

**For Evoxtral:** Don't just train on tagged transcriptions. Consider including:
- General ASR examples (plain transcription)
- Diverse audio conditions (clean, noisy, different speakers)
- Various text styles and lengths

---

## 2. Balanced Dataset Design for Structured Output Tasks

### Teaching When NOT to Apply Tags

This is a critical and under-researched area. The SSML annotation literature provides the closest parallels.

**From [SSML Prosody Control Research](https://arxiv.org/html/2508.17494v1):**
- Models consistently **under-generate** tags when not enough tagged examples exist
- But **over-generate** when training is tag-heavy
- The solution: systematic variation in tag density across training examples

**Recommended tag density distribution for Evoxtral training data:**

| Tag Density | % of Dataset | Description |
|-------------|-------------|-------------|
| None (0 tags) | 25-35% | Plain transcription, emotionally neutral audio |
| Light (1-2 tags) | 25-30% | Subtle emotion, single non-verbal |
| Medium (3-5 tags) | 25-30% | Multiple emotions, mixed delivery |
| Heavy (6+ tags) | 10-15% | Highly expressive, dramatic audio |

### Structured Output Quality

From [Databricks End-to-End Structured Extraction](https://community.databricks.com/t5/technical-blog/end-to-end-structured-extraction-with-llm-part-2-fine-tuning/ba-p/99900):
- Training data should be "structured, token-balanced, and metadata-tagged"
- For tagged output tasks, ensure the tokenizer properly handles your tag vocabulary
- Label masking (computing loss only on output tokens) is essential -- Evoxtral already plans this

---

## 3. Synthetic Data Quality and Diversity

### Best Practices from Research

**Quality filtering ([Eugene Yan's comprehensive guide](https://eugeneyan.com/writing/synthetic/)):**
- Use **ROUGE-L < 0.7** threshold against existing examples to ensure diversity (Self-Instruct method)
- Remove impossible instructions (e.g., referencing images for text-only models)
- Apply validation scoring: chain-of-thought + 5-point scale, average 3 scores per response
- **54% of synthetic samples having completely valid fields still improved performance by 33%** -- moderate imperfection is workable

**Diversity strategies:**
- **Iterative sampling**: Start with 8 seed examples, progressively incorporate generated ones
- **Template expansion**: Create 2+ alternative formulations for each task
- **Attribute conditioning**: Vary all controllable attributes systematically
- **Style variation**: Generate multiple styles (e.g., WRAP paper used easy/medium/hard/Q&A formats, achieving 3x training speedup with 1:1 real-to-synthetic ratio)

### Synthetic Data for Speech/Audio Tasks

**From [Optimized Synthetic Data for ASR](https://arxiv.org/html/2508.21631v1):**
- Cyclically iterate over speakers without replacement to maximize speaker diversity
- TTS and voice conversion systems are viable for ASR data augmentation
- Synthetic data lacks diversity in pitch, speed, and background noise compared to authentic audio

**From [Synthio Audio Classification](https://arxiv.org/html/2410.02056v1):**
- Enhancing consistency and diversity with a small-scale version of the target dataset significantly improves performance
- Data augmentations for acoustic diversity boost out-of-distribution generalization

### Stratified Sampling for Evoxtral

**Recommended stratification axes for the training dataset:**

| Axis | Categories | Rationale |
|------|-----------|-----------|
| Emotion type | excited, sad, angry, nervous, calm, frustrated | Balanced representation of all target emotions |
| Non-verbal sounds | laughs, sighs, gasps, clears throat, crying | Each sound type needs adequate coverage |
| Speaker gender | male, female, neutral | Prevent gender bias in emotion detection |
| Audio length | short (<10s), medium (10-30s), long (30s+) | Varied context window utilization |
| Tag density | none, light, medium, heavy (see table above) | Critical for preventing over/under-generation |
| Emotional valence | positive, negative, neutral | Prevent bias toward detecting only negative emotions |

**Speaker diversity from [Latent Mixup for Speech Recognition](https://arxiv.org/html/2511.20534):**
- Constrain pairings to match gender and dataset partition
- Maintain distribution characteristics across splits

---

## 4. Catastrophic Forgetting Prevention

### How Much Original-Task Data to Mix In

This is the most critical question for Evoxtral: how much plain ASR data to include
so that adding emotion tag capability doesn't degrade word-level transcription quality.

**Key finding from [Apple's Scaling Laws for Forgetting with Pretraining Data Injection](https://machinelearning.apple.com/research/scaling-laws):**
> Injecting as little as **1% of pretraining data** in the finetuning mixture prevents the model
> from forgetting the pretraining set.

**However, more nuanced findings from [Scaling Laws for Forgetting](https://arxiv.org/html/2401.05605v1):**
- Forgetting follows a **strong inverse linear relationship** with fine-tuning loss
- Forgetting increases as a **shifted power law** in both parameters finetuned and training steps
- Forgetting **cannot be avoided through early stopping** or varying parameter counts
- LoRA still suffers from forgetting, though less than full finetuning

**Concrete replay buffer recommendations by task type:**

| Task Type | Minimum Replay Buffer | Recommended Buffer | Source |
|-----------|----------------------|-------------------|--------|
| NLU tasks (classification, NLI) | 1-2% | 5% | Empirical study on catastrophic forgetting |
| Math/Code tasks | 5-10% | 15-20% | Same study |
| Structured output (like tags) | ~10% | 25-35% | Extrapolated from mixed training results |

**For Evoxtral specifically:**
- Since you're adding a **new structural capability** (tag generation) on top of an existing one (ASR), the risk is higher than simple domain adaptation
- **Recommended: 25-35% of training data should be plain ASR transcription** (no tags)
- This is supported by the mixed training study showing 1:1 ratio achieving equivalent base-task performance with only 0.7pp degradation

### The "Tax" of Adding New Capabilities

From the math finetuning study:
- **Math-only training**: Math accuracy went 3.1% -> 12.0%, but NLI dropped 81.0% -> 16.5% (catastrophic)
- **1:1 mixed training**: Math accuracy 12.0% (same!), NLI 86.2% (only 0.7pp drop from 86.9%)
- **Even 15:1 (93.8% new task)**: Original task maintained at 83.8% vs 86.9% baseline

**Bottom line**: With proper data mixing, the "tax" of adding tagged transcription capability
should be **less than 3% WER degradation** on plain transcription tasks, likely under 1% with
a 1:1 mix.

### LoRA-Specific Forgetting Mitigation

LoRA inherently reduces forgetting compared to full finetuning because:
- Fewer parameters are modified (lower rank = less forgetting)
- Base weights remain frozen
- The adapter can be merged or removed

However, the [scaling laws paper](https://arxiv.org/html/2401.05605v1) found LoRA still
exhibits forgetting that follows the same power law. The data mixing strategy remains essential
even with LoRA.

---

## 5. Class Imbalance in Tag/Label Finetuning

### The Problem for Evoxtral

Some tags will naturally be rarer than others:
- `[excited]` and `[laughs]` likely appear frequently
- `[gasps]`, `[stammers]`, `[clears throat]` are much rarer
- `[pause]` and emphasis (CAPS) are potentially in every example

### Balancing Strategies

**Three main approaches from [Class-Balanced Loss (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf):**

1. **Oversampling rare classes**: Simple but risks overfitting to repeated examples
2. **Undersampling common classes**: Loses valuable training signal
3. **Weighted loss**: Reweight by effective number of samples -- best theoretical approach

**Class-Balanced Loss formula:**
```
weight_i = (1 - beta) / (1 - beta^n_i)
```
where `n_i` = number of samples for class `i`, and `beta` is typically 0.9, 0.99, or 0.999.

**For generative models (Evoxtral's case), the [HuggingFace forum discussion](https://discuss.huggingface.co/t/handling-class-imbalance-when-finetuning-a-decoder-model-on-text-generation/173010) notes:**
- Weighted loss is harder to apply in token-level generation
- **Oversampling with variation is often more practical** for generative models
- Ensure rare tags appear in diverse contexts (different sentences, emotions, speakers)

### Recommended Strategy for Evoxtral

**Hybrid approach:**

1. **Stratified generation**: When creating synthetic training data, ensure minimum representation:
   - Each tag type should appear in at least 5-10% of tagged examples
   - Use the LLM script generator to specifically request rare tag scenarios

2. **Contextual oversampling**: For rare tags, generate multiple variations:
   - `[gasps]` in surprise context, fear context, excitement context
   - `[stammers]` in nervous context, angry context, confused context
   - Aim for 3-5x oversampling of the rarest tags relative to natural distribution

3. **Minimum tag frequency targets:**

| Tag Category | Minimum % of Tagged Examples | Natural Frequency | Oversampling Factor |
|-------------|-----------------------------|--------------------|---------------------|
| [excited], [sad], [angry] | 15-20% each | High | 1x (none) |
| [calm], [nervous], [frustrated] | 10-15% each | Medium | 1.5-2x |
| [laughs], [sighs] | 10-15% each | Medium-High | 1x |
| [gasps], [crying] | 8-12% each | Low | 2-3x |
| [whispers], [shouts] | 8-12% each | Low | 2-3x |
| [stammers], [clears throat] | 5-10% each | Very Low | 3-5x |
| [pause], CAPS emphasis | Present in 40-60% | Very High | 0.5x (undersample) |

---

## 6. Concrete Recommendations for Evoxtral Training Data

### Final Dataset Composition (for 800 total examples)

| Category | Count | Percentage | Description |
|----------|-------|------------|-------------|
| Heavily tagged | 80-120 | 10-15% | 6+ tags, dramatic/expressive audio |
| Medium tagged | 200-240 | 25-30% | 3-5 tags, moderate emotion |
| Lightly tagged | 200-240 | 25-30% | 1-2 tags, subtle emotion |
| Plain transcription | 240-280 | 30-35% | 0 tags, neutral delivery |

### Quality Checklist for Training Data

- [ ] Tag density varies naturally (not every sentence has a tag)
- [ ] All 15+ target tags appear in at least 40-80 examples
- [ ] Rare tags are oversampled 2-5x with diverse contexts
- [ ] 30-35% of examples are plain transcription (anti-hallucination)
- [ ] Speaker diversity: at least 6-8 distinct voices
- [ ] Audio length varies (short, medium, long segments)
- [ ] Emotional valence balanced (positive/negative/neutral)
- [ ] ROUGE-L between any two examples < 0.7 (diversity check)
- [ ] Tag positions vary within sentences (beginning, middle, end)
- [ ] Some examples have closely spaced tags, others widely spaced

### Training Configuration Notes

- **Epochs**: 1-2 (more increases forgetting risk)
- **LoRA rank**: Treat as hyperparameter; sweep [8, 16, 32, 64]
- **Learning rate**: Conservative (1e-5 to 5e-5 range)
- **Label masking**: Essential -- only compute loss on output tokens
- **Evaluation**: Track both WER (plain transcription quality) AND tag F1 simultaneously
- **Early stopping**: Monitor WER on a held-out plain transcription set; stop if it degrades >2%

---

## Sources

- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) - Sebastian Raschka
- [The Hallucination Tax of Reinforcement Finetuning](https://arxiv.org/html/2505.13988) - Negative example ratios
- [Unfamiliar Finetuning Examples Control How LLMs Hallucinate](https://arxiv.org/html/2403.05612v1)
- [Mitigating Catastrophic Forgetting via Mixed Training](https://arxiv.org/html/2512.13706) - Data replay ratios
- [Scaling Laws for Forgetting When Fine-Tuning LLMs](https://arxiv.org/html/2401.05605v1) - Power law relationships
- [Scaling Laws for Forgetting with Pretraining Data Injection](https://machinelearning.apple.com/research/scaling-laws) - Apple, 1% replay finding
- [Data Mixing Optimization for SFT](https://arxiv.org/html/2508.11953v1) - Cocktail effect
- [How to Generate and Use Synthetic Data for Finetuning](https://eugeneyan.com/writing/synthetic/) - Eugene Yan
- [On the Diversity of Synthetic Data](https://arxiv.org/html/2410.15226v2)
- [Data Diversity Matters for Robust Instruction Tuning](https://aclanthology.org/2024.findings-emnlp.195.pdf)
- [Class-Balanced Loss Based on Effective Number of Samples](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) - CVPR 2019
- [Improving French Synthetic Speech Quality via SSML](https://arxiv.org/html/2508.17494v1)
- [Towards Improved Speech Recognition through Synthetic Data](https://arxiv.org/html/2508.21631v1)
- [Efficient Fine-Tuning with LoRA Guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms) - Databricks
- [How to Fine-Tune Focus on Effective Datasets](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation/) - Meta
- [Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/) - Lilian Weng
