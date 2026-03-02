"""Run Evoxtral LoRA finetuning on Modal with GPU.

Usage:
    # First time setup:
    pip install modal
    modal setup
    modal secret create wandb-secret WANDB_API_KEY=your_key
    modal secret create huggingface-secret HF_TOKEN=hf_your_token

    # Upload training data to Modal volume:
    modal volume create evoxtral-data
    modal volume put evoxtral-data ./data/processed/ /processed/
    modal volume put evoxtral-data ./data/audio/ /audio/
    modal volume put evoxtral-data ./data/scripts/scripts.json /scripts/scripts.json

    # Run training:
    modal run scripts/train_modal.py

    # Download trained adapter:
    modal volume get evoxtral-output /evoxtral-lora ./model/evoxtral-lora/
"""

import os
import modal

# ── Modal Image ──────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
        "transformers==4.56.0",
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
        "mistral-common",
        "torchcodec",
        gpu="A10G",
    )
    .env({
        "HF_HUB_CACHE": "/cache/huggingface",
        "WANDB_LOG_MODEL": "end",
    })
)

# ── Modal App ────────────────────────────────────────────────────

app = modal.App("evoxtral-finetune", image=image)

# Persistent volumes
hf_cache = modal.Volume.from_name("evoxtral-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("evoxtral-data", create_if_missing=True)
output_vol = modal.Volume.from_name("evoxtral-output", create_if_missing=True)

VOLUMES = {
    "/cache/huggingface": hf_cache,
    "/data": data_vol,
    "/output": output_vol,
}


# ── Training Function ────────────────────────────────────────────

@app.function(
    gpu="A10G",
    volumes=VOLUMES,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=7200,  # 2 hours max
    memory=32768,
)
def train(
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    grad_accum: int = 8,
    lora_r: int = 64,
    lora_alpha: int = 128,
    push_to_hub: bool = True,
):
    """Run LoRA finetuning on Voxtral-Mini-3B."""
    import torch
    import wandb
    import json
    from pathlib import Path
    from datasets import load_from_disk, Audio
    from transformers import (
        VoxtralForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model

    MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
    OUTPUT_DIR = "/output/evoxtral-lora"
    HUB_MODEL_ID = "mistral-hackaton-2026/evoxtral-lora"
    PROMPT = "Transcribe this audio with expressive tags."

    # ── GPU info ──
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── W&B ──
    run = wandb.init(
        project="evoxtral",
        name=f"sft-lora-r{lora_r}-lr{learning_rate}-ep{num_epochs}",
        config={
            "model_id": MODEL_ID,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "neftune_noise_alpha": 5.0,
        },
        tags=["evoxtral", "voxtral", "lora", "sft", "neftune", "modal"],
    )

    # ── Log dataset artifact ──
    ds_artifact = wandb.Artifact("evoxtral-dataset", type="dataset")
    if Path("/data/scripts/scripts.json").exists():
        ds_artifact.add_file("/data/scripts/scripts.json", name="scripts.json")
    run.log_artifact(ds_artifact)

    # ── Load dataset ──
    ds = load_from_disk("/data/processed")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"Dataset: train={len(ds['train'])}, val={len(ds['validation'])}, test={len(ds['test'])}")
    wandb.log({
        "dataset/train_size": len(ds["train"]),
        "dataset/val_size": len(ds["validation"]),
    })

    # ── Load model ──
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Freeze Whisper audio encoder
    for param in model.audio_tower.parameters():
        param.requires_grad = False

    # ── LoRA ──
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
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
    # Uses apply_transcription_request() which accepts raw audio arrays,
    # then manually appends target text tokens and builds labels.
    class VoxtralDataCollator:
        def __init__(self, processor, model_id, max_text_len=512):
            self.processor = processor
            self.model_id = model_id
            self.max_text_len = max_text_len
            self.pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

        def __call__(self, examples):
            texts = [ex["tagged_text"] for ex in examples]
            audios = [ex["audio"]["array"] for ex in examples]

            # 1) Build prompt: [AUDIO]…[AUDIO] <transcribe> via processor
            prompt = self.processor.apply_transcription_request(
                language="en",
                model_id=self.model_id,
                audio=audios,
                format=["WAV"] * len(audios),
                return_tensors="pt",
            )
            # Keep audio features (input_features) for the model
            passthrough = {k: v for k, v in prompt.items()
                          if k not in ("input_ids", "attention_mask")}

            prompt_ids = prompt["input_ids"]        # [B, Lp]
            prompt_attn = prompt["attention_mask"]   # [B, Lp]
            B = prompt_ids.size(0)

            tok = self.processor.tokenizer
            # 2) Tokenize target texts
            text_tok = tok(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors=None,
            )
            text_ids_list = text_tok["input_ids"]

            # 3) Concatenate: input_ids = [PROMPT] + [TARGET] + [EOS]
            input_ids, attention_mask, labels = [], [], []
            for i in range(B):
                p_ids = prompt_ids[i].tolist()
                p_att = prompt_attn[i].tolist()
                t_ids = text_ids_list[i]

                ids = p_ids + t_ids + [tok.eos_token_id]
                attn = p_att + [1] * (len(t_ids) + 1)
                # Labels: mask prompt, learn only on target text + EOS
                lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

                input_ids.append(ids)
                attention_mask.append(attn)
                labels.append(lab)

            # 4) Pad to max length in batch
            max_len = max(len(x) for x in input_ids)

            def pad_to(seq, fill, L):
                return seq + [fill] * (L - len(seq))

            input_ids = [pad_to(x, self.pad_id, max_len) for x in input_ids]
            attention_mask = [pad_to(x, 0, max_len) for x in attention_mask]
            labels = [pad_to(x, -100, max_len) for x in labels]

            batch = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            # 5) Include audio features for the model
            for k, v in passthrough.items():
                batch[k] = v

            return batch

    collator = VoxtralDataCollator(processor, MODEL_ID)

    # ── Training args ──
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=5.0,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        push_to_hub=False,  # Push manually after training to avoid init_hf_repo error
    )

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )

    # ── Train ──
    print("Starting training...")
    train_result = trainer.train()

    # Log final metrics
    wandb.log({
        "train/final_loss": train_result.metrics.get("train_loss", 0),
        "train/runtime_seconds": train_result.metrics.get("train_runtime", 0),
    })

    # ── Save adapter ──
    print(f"Saving adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Log adapter as W&B artifact
    adapter_artifact = wandb.Artifact(
        "evoxtral-lora-adapter",
        type="model",
        metadata={
            "wandb.base_model": MODEL_ID,
            "lora_rank": lora_r,
            "lora_alpha": lora_alpha,
            "hub_model_id": HUB_MODEL_ID,
        },
    )
    adapter_artifact.add_dir(OUTPUT_DIR)
    logged = run.log_artifact(adapter_artifact)
    run.link_artifact(logged, target_path="wandb-registry-model/evoxtral-lora")

    # Push to Hub using HfApi directly
    if push_to_hub:
        from huggingface_hub import HfApi
        print(f"Pushing to HuggingFace Hub: {HUB_MODEL_ID}")
        try:
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            api.create_repo(HUB_MODEL_ID, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=HUB_MODEL_ID,
                repo_type="model",
                commit_message=f"LoRA adapter: r={lora_r}, lr={learning_rate}, ep={num_epochs}",
            )
            print(f"Successfully pushed to {HUB_MODEL_ID}")
        except Exception as e:
            print(f"WARNING: Hub push failed: {e}. Adapter saved locally at {OUTPUT_DIR}")

    output_vol.commit()
    wandb.finish()
    print("Training complete!")
    return train_result.metrics


# ── Evaluation Function ──────────────────────────────────────────

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
def evaluate(adapter_path: str = "/output/evoxtral-lora", eval_name: str = "finetuned"):
    """Run Evoxtral-Bench evaluation (base vs finetuned)."""
    import torch
    import wandb
    import weave
    import json
    from pathlib import Path
    from jiwer import wer as compute_wer, cer as compute_cer_jiwer
    from datasets import load_from_disk, Audio
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    import re
    from collections import Counter

    MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
    PROMPT = "Transcribe this audio with expressive tags."

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

    # Load test set
    ds = load_from_disk("/data/processed")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    test_ds = ds["test"]

    def compute_cer(ref, hyp):
        """Character Error Rate."""
        if not ref.strip():
            return 0.0
        return compute_cer_jiwer(ref, hyp)

    def run_model_eval(model, processor, model_name):
        model.eval()
        results = {
            "wer": [], "cer": [],
            "tag_f1": [], "tag_precision": [], "tag_recall": [],
            "tag_hallucination_rate": [],
            "emphasis_f1": [],
        }
        per_tag_tp = Counter()  # true positives per tag type
        per_tag_fp = Counter()  # false positives (predicted but not in ref)
        per_tag_fn = Counter()  # false negatives (in ref but not predicted)
        all_predictions = []

        for i in range(min(len(test_ds), 50)):  # eval on up to 50 samples
            row = test_ds[i]
            tagged_text = row["tagged_text"]

            audio_array = row["audio"]["array"]

            inputs = processor.apply_transcription_request(
                language="en",
                audio=[audio_array],
                format=["WAV"],
                model_id=MODEL_ID,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            prediction = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

            # --- Text metrics ---
            ref_plain = strip_tags(tagged_text)
            pred_plain = strip_tags(prediction)
            if ref_plain.strip():
                results["wer"].append(compute_wer(ref_plain, pred_plain))
                results["cer"].append(compute_cer(ref_plain, pred_plain))

            # --- Tag metrics (precision, recall, F1) ---
            pred_tags = Counter(extract_tags(prediction))
            ref_tags = Counter(extract_tags(tagged_text))

            common = pred_tags & ref_tags
            tp = sum(common.values())
            fp = sum(pred_tags.values()) - tp
            fn = sum(ref_tags.values()) - tp

            prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not ref_tags else 0.0)
            rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not pred_tags else 0.0)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else (1.0 if not ref_tags and not pred_tags else 0.0)

            results["tag_precision"].append(prec)
            results["tag_recall"].append(rec)
            results["tag_f1"].append(f1)

            # Tag hallucination rate
            ref_tag_set = set(ref_tags.keys())
            hallucinated = sum(v for k, v in pred_tags.items() if k not in ref_tag_set)
            hall_rate = hallucinated / sum(pred_tags.values()) if pred_tags else 0.0
            results["tag_hallucination_rate"].append(hall_rate)

            # Per-tag breakdown
            for tag in set(list(pred_tags.keys()) + list(ref_tags.keys())):
                matched = min(pred_tags.get(tag, 0), ref_tags.get(tag, 0))
                per_tag_tp[tag] += matched
                per_tag_fp[tag] += max(0, pred_tags.get(tag, 0) - matched)
                per_tag_fn[tag] += max(0, ref_tags.get(tag, 0) - matched)

            # Emphasis F1
            pred_emph = Counter([m.group(0).lower() for m in re.finditer(r'\b[A-Z]{2,}\b', prediction)])
            ref_emph = Counter([m.group(0).lower() for m in re.finditer(r'\b[A-Z]{2,}\b', tagged_text)])
            emph_common = sum((pred_emph & ref_emph).values())
            emph_total_p = sum(pred_emph.values())
            emph_total_r = sum(ref_emph.values())
            if emph_total_p == 0 and emph_total_r == 0:
                emph_f1 = 1.0
            elif emph_total_p == 0 or emph_total_r == 0:
                emph_f1 = 0.0
            else:
                ep = emph_common / emph_total_p
                er = emph_common / emph_total_r
                emph_f1 = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0
            results["emphasis_f1"].append(emph_f1)

            # Store prediction for W&B table
            all_predictions.append({
                "sample_idx": i,
                "reference": tagged_text,
                "prediction": prediction,
                "wer": results["wer"][-1] if results["wer"] else None,
                "tag_f1": f1,
                "tag_hallucination_rate": hall_rate,
            })

            if i < 5:
                print(f"\n[{model_name} Sample {i}]")
                print(f"  Reference: {tagged_text[:100]}...")
                print(f"  Predicted: {prediction[:100]}...")

        # Compute averages
        def avg(lst):
            return sum(lst) / max(len(lst), 1)

        avg_metrics = {f"avg_{k}": avg(v) for k, v in results.items()}

        # Per-tag F1 breakdown
        per_tag_f1 = {}
        for tag in sorted(per_tag_tp.keys() | per_tag_fp.keys() | per_tag_fn.keys()):
            tp = per_tag_tp[tag]
            fp = per_tag_fp[tag]
            fn = per_tag_fn[tag]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_tag_f1[tag] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                               "support": tp + fn}

        avg_metrics["per_tag_f1"] = per_tag_f1
        avg_metrics["predictions"] = all_predictions

        print(f"\n{model_name} Results:")
        print(f"  WER:                  {avg_metrics['avg_wer']:.4f}")
        print(f"  CER:                  {avg_metrics['avg_cer']:.4f}")
        print(f"  Tag F1:               {avg_metrics['avg_tag_f1']:.4f}")
        print(f"  Tag Precision:        {avg_metrics['avg_tag_precision']:.4f}")
        print(f"  Tag Recall:           {avg_metrics['avg_tag_recall']:.4f}")
        print(f"  Tag Hallucination:    {avg_metrics['avg_tag_hallucination_rate']:.4f}")
        print(f"  Emphasis F1:          {avg_metrics['avg_emphasis_f1']:.4f}")
        print(f"\n  Per-tag breakdown:")
        for tag, m in sorted(per_tag_f1.items(), key=lambda x: -x[1]["support"]):
            print(f"    [{tag}]: F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} (n={m['support']})")

        return avg_metrics

    # ── Evaluate base model ──
    wandb.init(project="evoxtral", name="eval-base", job_type="evaluation", tags=["eval", "base"])
    print("Loading base model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
    )
    base_results = run_model_eval(base_model, processor, "BASE")
    wandb.log({
        "eval/wer": base_results["avg_wer"],
        "eval/cer": base_results["avg_cer"],
        "eval/tag_f1": base_results["avg_tag_f1"],
        "eval/tag_precision": base_results["avg_tag_precision"],
        "eval/tag_recall": base_results["avg_tag_recall"],
        "eval/tag_hallucination_rate": base_results["avg_tag_hallucination_rate"],
        "eval/emphasis_f1": base_results["avg_emphasis_f1"],
    })
    # Log per-tag breakdown as table
    tag_table = wandb.Table(columns=["tag", "f1", "precision", "recall", "support"])
    for tag, m in base_results["per_tag_f1"].items():
        tag_table.add_data(tag, m["f1"], m["precision"], m["recall"], m["support"])
    wandb.log({"eval/per_tag_breakdown": tag_table})
    # Log predictions table
    pred_table = wandb.Table(columns=["idx", "reference", "prediction", "wer", "tag_f1", "hallucination_rate"])
    for p in base_results["predictions"]:
        pred_table.add_data(p["sample_idx"], p["reference"], p["prediction"],
                           p["wer"], p["tag_f1"], p["tag_hallucination_rate"])
    wandb.log({"eval/predictions": pred_table})
    wandb.finish()

    # Free base model memory
    del base_model
    torch.cuda.empty_cache()

    # ── Evaluate finetuned model ──
    if Path(adapter_path).exists():
        wandb.init(project="evoxtral", name=f"eval-{eval_name}", job_type="evaluation", tags=["eval", eval_name])
        print("Loading finetuned model...")
        base_model = VoxtralForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        )
        ft_model = PeftModel.from_pretrained(base_model, adapter_path)
        ft_results = run_model_eval(ft_model, processor, "FINETUNED")
        wandb.log({
            "eval/wer": ft_results["avg_wer"],
            "eval/cer": ft_results["avg_cer"],
            "eval/tag_f1": ft_results["avg_tag_f1"],
            "eval/tag_precision": ft_results["avg_tag_precision"],
            "eval/tag_recall": ft_results["avg_tag_recall"],
            "eval/tag_hallucination_rate": ft_results["avg_tag_hallucination_rate"],
            "eval/emphasis_f1": ft_results["avg_emphasis_f1"],
        })
        # Log per-tag breakdown as table
        tag_table = wandb.Table(columns=["tag", "f1", "precision", "recall", "support"])
        for tag, m in ft_results["per_tag_f1"].items():
            tag_table.add_data(tag, m["f1"], m["precision"], m["recall"], m["support"])
        wandb.log({"eval/per_tag_breakdown": tag_table})
        # Log predictions table
        pred_table = wandb.Table(columns=["idx", "reference", "prediction", "wer", "tag_f1", "hallucination_rate"])
        for p in ft_results["predictions"]:
            pred_table.add_data(p["sample_idx"], p["reference"], p["prediction"],
                               p["wer"], p["tag_f1"], p["tag_hallucination_rate"])
        wandb.log({"eval/predictions": pred_table})
        wandb.finish()

        print(f"\n{'='*60}")
        print(f"COMPARISON: Base vs Finetuned")
        print(f"{'='*60}")
        print(f"WER:                {base_results['avg_wer']:.4f} → {ft_results['avg_wer']:.4f}")
        print(f"CER:                {base_results['avg_cer']:.4f} → {ft_results['avg_cer']:.4f}")
        print(f"Tag F1:             {base_results['avg_tag_f1']:.4f} → {ft_results['avg_tag_f1']:.4f}")
        print(f"Tag Precision:      {base_results['avg_tag_precision']:.4f} → {ft_results['avg_tag_precision']:.4f}")
        print(f"Tag Recall:         {base_results['avg_tag_recall']:.4f} → {ft_results['avg_tag_recall']:.4f}")
        print(f"Tag Hallucination:  {base_results['avg_tag_hallucination_rate']:.4f} → {ft_results['avg_tag_hallucination_rate']:.4f}")
        print(f"Emphasis F1:        {base_results['avg_emphasis_f1']:.4f} → {ft_results['avg_emphasis_f1']:.4f}")
    else:
        print(f"No adapter found at {adapter_path}, skipping finetuned eval")
        ft_results = None

    output_vol.commit()
    return {"base": base_results, "finetuned": ft_results}


# ── Local Entrypoint ─────────────────────────────────────────────

@app.local_entrypoint()
def main(
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 2,
    push_to_hub: bool = True,
    eval_only: bool = False,
    eval_rl: bool = False,
):
    if eval_rl:
        results = evaluate.remote(adapter_path="/output/evoxtral-rl", eval_name="rl")
        print(results)
    elif eval_only:
        results = evaluate.remote()
        print(results)
    else:
        metrics = train.remote(
            num_epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            push_to_hub=push_to_hub,
        )
        print(f"Training metrics: {metrics}")

        print("\nRunning evaluation...")
        results = evaluate.remote()
        print(f"Eval results: {results}")
