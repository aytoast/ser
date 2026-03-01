## core components
- **base model:** Ethostral (fine-tuned Mistral).
- **tracking and evaluation:** weights and biases.
- **platform:** hugging face for model and adapter hosting.

## architecture
1. **process:** audio is streamed to the fine-tuned mistral voxtral endpoint for simultaneous automatic speech recognition and emotion classification.
2. **output format:** transcription output uses interleaved text and emotional metadata tags.
3. **frontend:** Next.js application utilizing shadcn UI (Maia style) and Phosphor icons for the interactive dashboard.

## integration points
- **weights and biases weave:** used for tracing the recognition pipeline.
- **hugging face hub:** serves as the repository for fine-tuned weights and dataset storage.
- **shadcn ui:** component library with maia theme.
- **phosphor icons:** primary iconography set.

## performance metrics
- word error rate for transcription quality.
- f1 score for emotion detection accuracy.

## benchmarking and evals
- **IEMOCAP:** Evaluation of categorical and dimensional (Valence/Arousal/Dominance) accuracy.
- **RAVDESS:** Benchmarking of prosodic feature mapping and speech rate accuracy.
- **SUSAS:** Evaluation of stress detection reliability under varied acoustic conditions.
- **MDPE:** Assessment of deception-related emotional leakage detection.
- **Weights & Biases Weave:** Used for tracking eval traces and scoring pipeline performance.