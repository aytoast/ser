"""Evoxtral Self-Improving Workflow — Auto Hill-Climbing Optimization.

Automatically runs multiple training experiments with different hyperparameters,
evaluates each, and selects the best configuration. All tracked via W&B.

This is designed for the W&B mini challenge: "Best self-improving workflow"

Usage:
    modal run scripts/auto_optimize.py
    modal run scripts/auto_optimize.py --max-trials 5
"""

import modal
import os
import json
import itertools

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
        "transformers>=4.56.0",
        "datasets>=2.14.0",
        "accelerate>=1.0.0",
        "peft>=0.13.0",
        "wandb>=0.18.0",
        "weave>=0.50.0",
        "jiwer>=3.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "huggingface_hub",
        "safetensors",
        "sentencepiece",
        gpu="A10G",
    )
    .env({
        "HF_HUB_CACHE": "/cache/huggingface",
        "WANDB_LOG_MODEL": "end",
    })
)

app = modal.App("evoxtral-auto-optimize", image=image)

hf_cache = modal.Volume.from_name("evoxtral-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("evoxtral-data", create_if_missing=True)
output_vol = modal.Volume.from_name("evoxtral-output", create_if_missing=True)

VOLUMES = {
    "/cache/huggingface": hf_cache,
    "/data": data_vol,
    "/output": output_vol,
}

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
HUB_MODEL_ID = "mistral-hackaton-2026/evoxtral-lora"
PROMPT = "Transcribe this audio with expressive tags."

# ── Hyperparameter Search Space ──────────────────────────────────

SEARCH_SPACE = {
    "lora_r": [16, 32, 64],
    "lora_alpha_ratio": [2],       # alpha = r * ratio
    "learning_rate": [1e-4, 2e-4, 5e-4],
    "neftune_noise_alpha": [0.0, 5.0],
    "num_epochs": [2, 3],
}

def generate_configs(max_trials: int = 8) -> list[dict]:
    """Generate hyperparameter configs to try. Uses grid subset."""
    keys = list(SEARCH_SPACE.keys())
    all_combos = list(itertools.product(*SEARCH_SPACE.values()))

    configs = []
    # Take evenly spaced subset if too many
    step = max(1, len(all_combos) // max_trials)
    for i in range(0, len(all_combos), step):
        if len(configs) >= max_trials:
            break
        combo = dict(zip(keys, all_combos[i]))
        combo["lora_alpha"] = combo["lora_r"] * combo.pop("lora_alpha_ratio")
        configs.append(combo)

    return configs


# ── Single Trial: Train + Eval ───────────────────────────────────

@app.function(
    gpu="A10G",
    volumes=VOLUMES,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=7200,
    memory=32768,
)
def run_trial(trial_id: int, config: dict) -> dict:
    """Run a single training trial and evaluate it."""
    import torch
    import wandb
    import re
    from pathlib import Path
    from collections import Counter
    from datasets import load_from_disk, Audio
    from transformers import (
        VoxtralForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from jiwer import wer as compute_wer

    trial_name = f"trial-{trial_id:02d}-r{config['lora_r']}-lr{config['learning_rate']}"
    output_dir = f"/output/trials/{trial_name}"

    # ── Metric helpers ──
    def extract_tags(text):
        return [m.group(1).lower() for m in re.finditer(r'\[([^\]]+)\]', text)]

    def strip_tags(text):
        text = re.sub(r'\[[^\]]+\]\s*', '', text)
        text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group(0).lower(), text)
        return text.strip()

    def compute_tag_f1(pred, ref):
        pred_tags = Counter(extract_tags(pred))
        ref_tags = Counter(extract_tags(ref))
        if not ref_tags and not pred_tags:
            return 1.0
        if not ref_tags or not pred_tags:
            return 0.0
        common = sum((pred_tags & ref_tags).values())
        prec = common / sum(pred_tags.values())
        rec = common / sum(ref_tags.values())
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ── W&B ──
    run = wandb.init(
        project="evoxtral",
        name=trial_name,
        group="auto-optimize",
        config={
            "trial_id": trial_id,
            **config,
            "model_id": MODEL_ID,
        },
        tags=["auto-optimize", "trial", f"r{config['lora_r']}", f"lr{config['learning_rate']}"],
    )

    print(f"\n{'='*60}")
    print(f"TRIAL {trial_id}: {trial_name}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*60}")

    # ── Load data ──
    ds = load_from_disk("/data/processed")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # ── Load model ──
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    for param in model.encoder.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "multi_modal_projector.linear_1",
            "multi_modal_projector.linear_2",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Data Collator ──
    class Collator:
        def __init__(self, proc, max_len=2048):
            self.proc = proc
            self.max_len = max_len

        def __call__(self, examples):
            convs, targets = [], []
            for ex in examples:
                conv = [
                    {"role": "user", "content": [
                        {"type": "audio", "audio": ex["audio"]["array"]},
                        {"type": "text", "text": PROMPT},
                    ]},
                    {"role": "assistant", "content": ex["tagged_text"]},
                ]
                convs.append(conv)
                targets.append(ex["tagged_text"])

            inputs = self.proc.apply_chat_template(
                convs, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt",
            )
            labels = inputs["input_ids"].clone()
            if self.proc.tokenizer.pad_token_id is not None:
                labels[labels == self.proc.tokenizer.pad_token_id] = -100
            for i, t in enumerate(targets):
                t_ids = self.proc.tokenizer.encode(t, add_special_tokens=False)
                seq_len = (labels[i] != -100).sum().item()
                prompt_len = seq_len - len(t_ids)
                if prompt_len > 0:
                    labels[i, :prompt_len] = -100
            inputs["labels"] = labels
            return inputs

    # ── Train ──
    neftune = config.get("neftune_noise_alpha", 5.0)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=neftune if neftune > 0 else None,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=Collator(processor),
    )

    train_result = trainer.train()
    final_loss = train_result.metrics.get("train_loss", float("inf"))

    # Save adapter
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # ── Evaluate ──
    print(f"\nEvaluating trial {trial_id}...")
    model.eval()
    test_ds = ds["test"]
    wer_scores, f1_scores = [], []

    for i in range(min(len(test_ds), 30)):
        row = test_ds[i]
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio": row["audio"]["array"]},
                {"type": "text", "text": PROMPT},
            ]},
        ]
        inputs = processor.apply_chat_template([conversation], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        pred = processor.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        ref_plain = strip_tags(row["tagged_text"])
        pred_plain = strip_tags(pred)
        if ref_plain.strip():
            wer_scores.append(compute_wer(ref_plain, pred_plain))
        f1_scores.append(compute_tag_f1(pred, row["tagged_text"]))

    avg_wer = sum(wer_scores) / max(len(wer_scores), 1)
    avg_f1 = sum(f1_scores) / max(len(f1_scores), 1)
    # Combined score: lower is better (WER) + higher is better (F1)
    combined_score = avg_f1 - avg_wer  # higher = better

    # Log results
    trial_results = {
        "trial_id": trial_id,
        "config": config,
        "train_loss": final_loss,
        "eval_wer": avg_wer,
        "eval_tag_f1": avg_f1,
        "combined_score": combined_score,
        "output_dir": output_dir,
    }

    wandb.log({
        "eval/wer": avg_wer,
        "eval/tag_f1": avg_f1,
        "eval/combined_score": combined_score,
        "train/final_loss": final_loss,
    })

    # Log adapter artifact
    artifact = wandb.Artifact(
        f"evoxtral-trial-{trial_id}", type="model",
        metadata={"wandb.base_model": MODEL_ID, **config, "eval_wer": avg_wer, "eval_tag_f1": avg_f1},
    )
    artifact.add_dir(output_dir)
    run.log_artifact(artifact)

    wandb.finish()
    output_vol.commit()

    print(f"\nTrial {trial_id} complete: WER={avg_wer:.4f}, Tag F1={avg_f1:.4f}, Score={combined_score:.4f}")
    return trial_results


# ── Orchestrator: Pick Best + Push ───────────────────────────────

@app.function(
    gpu="A10G",
    volumes=VOLUMES,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=3600,
    memory=32768,
)
def promote_best(best_trial: dict):
    """Promote the best trial's adapter to HF Hub and W&B registry."""
    import wandb
    from pathlib import Path
    from huggingface_hub import HfApi

    trial_dir = best_trial["output_dir"]
    if not Path(trial_dir).exists():
        print(f"ERROR: {trial_dir} not found")
        return

    run = wandb.init(
        project="evoxtral",
        name="promote-best",
        job_type="promotion",
        config=best_trial,
        tags=["auto-optimize", "best", "promotion"],
    )

    # Log best adapter as final artifact
    artifact = wandb.Artifact(
        "evoxtral-lora-best", type="model",
        metadata={
            "wandb.base_model": MODEL_ID,
            **best_trial["config"],
            "eval_wer": best_trial["eval_wer"],
            "eval_tag_f1": best_trial["eval_tag_f1"],
            "combined_score": best_trial["combined_score"],
        },
    )
    artifact.add_dir(trial_dir)
    logged = run.log_artifact(artifact)
    run.link_artifact(logged, target_path="wandb-registry-model/evoxtral-lora")

    # Push best to HF Hub
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_folder(
        folder_path=trial_dir,
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        commit_message=f"Auto-optimized best trial: WER={best_trial['eval_wer']:.4f}, Tag F1={best_trial['eval_tag_f1']:.4f}",
    )
    print(f"Best adapter pushed to HF Hub: {HUB_MODEL_ID}")

    # Log summary table
    wandb.log({
        "best/trial_id": best_trial["trial_id"],
        "best/wer": best_trial["eval_wer"],
        "best/tag_f1": best_trial["eval_tag_f1"],
        "best/combined_score": best_trial["combined_score"],
        "best/lora_r": best_trial["config"]["lora_r"],
        "best/learning_rate": best_trial["config"]["learning_rate"],
    })

    wandb.finish()
    print("Promotion complete!")


# ── Main Entrypoint ──────────────────────────────────────────────

@app.local_entrypoint()
def main(max_trials: int = 6):
    """Run the self-improving optimization loop."""
    configs = generate_configs(max_trials)
    print(f"\n{'='*60}")
    print(f"EVOXTRAL AUTO-OPTIMIZE: {len(configs)} trials")
    print(f"{'='*60}")
    for i, cfg in enumerate(configs):
        print(f"  Trial {i}: r={cfg['lora_r']}, lr={cfg['learning_rate']}, "
              f"alpha={cfg['lora_alpha']}, neftune={cfg.get('neftune_noise_alpha', 5.0)}, "
              f"epochs={cfg.get('num_epochs', 3)}")

    # Run trials sequentially (GPU memory constraint)
    all_results = []
    for i, cfg in enumerate(configs):
        print(f"\n--- Starting Trial {i}/{len(configs)} ---")
        result = run_trial.remote(trial_id=i, config=cfg)
        all_results.append(result)
        print(f"Trial {i} result: WER={result['eval_wer']:.4f}, F1={result['eval_tag_f1']:.4f}")

    # Find best
    best = max(all_results, key=lambda r: r["combined_score"])
    print(f"\n{'='*60}")
    print(f"BEST TRIAL: #{best['trial_id']}")
    print(f"  Config: r={best['config']['lora_r']}, lr={best['config']['learning_rate']}")
    print(f"  WER: {best['eval_wer']:.4f}")
    print(f"  Tag F1: {best['eval_tag_f1']:.4f}")
    print(f"  Combined: {best['combined_score']:.4f}")
    print(f"{'='*60}")

    # Promote best to HF Hub + W&B registry
    print("\nPromoting best trial...")
    promote_best.remote(best)

    # Save results locally
    with open("optimization_results.json", "w") as f:
        json.dump({"trials": all_results, "best": best}, f, indent=2)
    print(f"\nResults saved to optimization_results.json")
