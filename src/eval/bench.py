"""Evoxtral-Bench: 3-layer evaluation with W&B Weave tracing.

Layer 1: Text quality (WER on plain text)
Layer 2: Tag classification (F1 on tag extraction + emphasis)
Layer 3: Round-trip TTS quality (optional)

All evaluations are traced via W&B Weave for full reproducibility.
"""

import asyncio
import json
import torch
import wandb
import weave
from pathlib import Path
from datasets import load_from_disk, Audio
from jiwer import wer as compute_wer
from dotenv import load_dotenv

from .tag_metrics import tag_f1, emphasis_f1, tag_hallucination_rate, strip_tags

load_dotenv()


# ── Weave Scorer Functions ──────────────────────────────────────────────

@weave.op()
def wer_scorer(output: dict, expected: str) -> dict:
    """Layer 1: Word Error Rate on plain text (tags stripped)."""
    pred_plain = strip_tags(output["prediction"])
    ref_plain = strip_tags(expected)
    score = compute_wer(ref_plain, pred_plain) if ref_plain.strip() else 0.0
    return {"wer": score}


@weave.op()
def tag_f1_scorer(output: dict, expected: str) -> dict:
    """Layer 2a: Tag extraction F1."""
    return tag_f1(output["prediction"], expected)


@weave.op()
def emphasis_f1_scorer(output: dict, expected: str) -> dict:
    """Layer 2b: Emphasis (CAPS) F1."""
    return emphasis_f1(output["prediction"], expected)


@weave.op()
def hallucination_scorer(output: dict, expected: str) -> dict:
    """Layer 2c: Tag hallucination rate."""
    rate = tag_hallucination_rate(output["prediction"], expected)
    return {"tag_hallucination_rate": rate}


# ── Model Wrapper ───────────────────────────────────────────────────────

class EvoxtralModel(weave.Model):
    """Wraps Voxtral inference for Weave evaluation."""

    model_id: str = "mistralai/Voxtral-Mini-3B-2507"
    adapter_path: str | None = None
    _model: object = None
    _processor: object = None

    class Config:
        arbitrary_types_allowed = True

    def _load(self):
        if self._model is not None:
            return

        from transformers import VoxtralForConditionalGeneration, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)

        if self.adapter_path:
            from peft import PeftModel
            base = VoxtralForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto",
            )
            self._model = PeftModel.from_pretrained(base, self.adapter_path)
        else:
            self._model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto",
            )
        self._model.eval()

    @weave.op()
    def predict(self, audio_path: str) -> dict:
        """Run inference on a single audio file."""
        self._load()

        import librosa
        audio_array, sr = librosa.load(audio_path, sr=16000)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": "Transcribe this audio with expressive tags."},
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            [conversation],
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Decode only the generated tokens (skip input)
        input_len = inputs["input_ids"].shape[1]
        prediction = self._processor.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True,
        )

        return {"prediction": prediction}


# ── Evaluation Runner ───────────────────────────────────────────────────

def build_eval_dataset(
    dataset_path: str = "data/processed",
    split: str = "test",
    manifest_path: str = "data/audio/manifest.json",
) -> list[dict]:
    """Build evaluation examples from the processed dataset."""
    ds = load_from_disk(dataset_path)
    test_ds = ds[split]

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build index from tagged_text to audio_path
    audio_lookup = {m["tagged_text"]: m["audio_path"] for m in manifest if m.get("synthesis_success")}

    examples = []
    for i in range(len(test_ds)):
        row = test_ds[i]
        tagged_text = row["tagged_text"]
        audio_path = audio_lookup.get(tagged_text)
        if audio_path and Path(audio_path).exists():
            examples.append({
                "audio_path": audio_path,
                "expected": tagged_text,
            })

    return examples


def run_eval(
    adapter_path: str | None = None,
    dataset_path: str = "data/processed",
    split: str = "test",
    wandb_project: str = "evoxtral",
):
    """Run full Evoxtral-Bench evaluation with Weave tracing."""
    weave.init(f"{wandb_project}")

    # Build eval dataset
    examples = build_eval_dataset(dataset_path, split)
    print(f"Evaluation examples: {len(examples)}")

    eval_dataset = weave.Dataset(
        name=f"evoxtral-bench-{split}",
        rows=examples,
    )

    # Create model (base or finetuned)
    model_name = "evoxtral-finetuned" if adapter_path else "voxtral-base"
    model = EvoxtralModel(
        adapter_path=adapter_path,
    )

    # Run evaluation
    evaluation = weave.Evaluation(
        dataset=eval_dataset,
        scorers=[wer_scorer, tag_f1_scorer, emphasis_f1_scorer, hallucination_scorer],
        evaluation_name=f"evoxtral-bench-{model_name}",
    )

    results = asyncio.run(evaluation.evaluate(model))

    # Also log summary to W&B run for easy comparison
    run = wandb.init(
        project=wandb_project,
        name=f"eval-{model_name}",
        job_type="evaluation",
        tags=["eval", model_name],
    )

    wandb.log({
        "eval/model": model_name,
        "eval/adapter_path": adapter_path or "none",
        "eval/num_examples": len(examples),
        "eval/split": split,
    })

    # Log per-metric summary
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, (int, float)):
                wandb.log({f"eval/{key}": value})

    wandb.finish()
    print(f"Evaluation complete for {model_name}")
    return results


def compare_base_vs_finetuned(
    adapter_path: str = "model/evoxtral-lora",
    dataset_path: str = "data/processed",
    wandb_project: str = "evoxtral",
):
    """Run eval on both base and finetuned model for comparison."""
    print("=" * 50)
    print("EVALUATING BASE MODEL")
    print("=" * 50)
    base_results = run_eval(
        adapter_path=None,
        dataset_path=dataset_path,
        wandb_project=wandb_project,
    )

    print("\n" + "=" * 50)
    print("EVALUATING FINETUNED MODEL")
    print("=" * 50)
    ft_results = run_eval(
        adapter_path=adapter_path,
        dataset_path=dataset_path,
        wandb_project=wandb_project,
    )

    return {"base": base_results, "finetuned": ft_results}


if __name__ == "__main__":
    import sys
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    if adapter == "compare":
        compare_base_vs_finetuned()
    else:
        run_eval(adapter_path=adapter)
