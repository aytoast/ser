"""Evoxtral RL — Rejection sampling + SFT on best completions (RAFT).

Follows the GRPO-for-ASR approach (arxiv:2509.01939) simplified for hackathon:
1. Generate N completions per training sample (with sampling)
2. Score each with rule-based reward (WER + Tag F1 + hallucination penalty)
3. Keep best completion per sample
4. SFT on the curated high-quality dataset (1 epoch, lower LR)

Usage:
    modal run scripts/rl_modal.py
    modal run scripts/rl_modal.py --num-samples 4 --lr 5e-5
"""

import os
import modal

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
        "jiwer>=3.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "huggingface_hub",
        "safetensors",
        "sentencepiece",
        "mistral-common",
        gpu="A10G",
    )
    .env({
        "HF_HUB_CACHE": "/cache/huggingface",
    })
)

app = modal.App("evoxtral-rl", image=image)

hf_cache = modal.Volume.from_name("evoxtral-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("evoxtral-data", create_if_missing=True)
output_vol = modal.Volume.from_name("evoxtral-output", create_if_missing=True)

VOLUMES = {
    "/cache/huggingface": hf_cache,
    "/data": data_vol,
    "/output": output_vol,
}

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
SFT_ADAPTER = "/output/evoxtral-lora"
RL_OUTPUT = "/output/evoxtral-rl"


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
def generate_and_score(num_samples: int = 4, temperature: float = 0.7):
    """Step 1: Generate N completions per sample and score them."""
    import torch
    import json
    import re
    from pathlib import Path
    from collections import Counter
    from jiwer import wer as compute_wer
    from datasets import load_from_disk, Audio
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Reward helpers ---
    def extract_tags(text):
        return [m.group(1).lower() for m in re.finditer(r'\[([^\]]+)\]', text)]

    def strip_tags(text):
        text = re.sub(r'\[[^\]]+\]\s*', '', text)
        text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group(0).lower(), text)
        return text.strip()

    def compute_reward(prediction, reference):
        """Rule-based reward: WER accuracy + Tag F1 - hallucination penalty."""
        # WER component (accuracy = 1 - WER)
        ref_plain = strip_tags(reference)
        pred_plain = strip_tags(prediction)
        if ref_plain.strip():
            wer_score = compute_wer(ref_plain, pred_plain)
            wer_accuracy = max(0.0, 1.0 - wer_score)
        else:
            wer_accuracy = 1.0

        # Tag F1 component
        pred_tags = Counter(extract_tags(prediction))
        ref_tags = Counter(extract_tags(reference))
        if not ref_tags and not pred_tags:
            tag_f1 = 1.0
            hall_rate = 0.0
        elif not ref_tags or not pred_tags:
            tag_f1 = 0.0
            hall_rate = 1.0 if pred_tags else 0.0
        else:
            common = sum((pred_tags & ref_tags).values())
            prec = common / sum(pred_tags.values())
            rec = common / sum(ref_tags.values())
            tag_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            # Hallucination rate
            ref_set = set(ref_tags.keys())
            hallucinated = sum(v for k, v in pred_tags.items() if k not in ref_set)
            hall_rate = hallucinated / sum(pred_tags.values())

        # Combined reward
        reward = 0.4 * wer_accuracy + 0.4 * tag_f1 + 0.2 * (1.0 - hall_rate)
        return reward, {"wer_accuracy": wer_accuracy, "tag_f1": tag_f1, "hall_rate": hall_rate}

    # --- Load model ---
    print("Loading SFT model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)
    model.eval()
    print(f"Model loaded on {model.device}")

    # --- Load training data ---
    ds = load_from_disk("/data/processed")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    train_ds = ds["train"]

    print(f"Generating {num_samples} completions per sample for {len(train_ds)} training examples...")

    curated_data = []
    total_reward = 0.0

    for i in range(len(train_ds)):
        row = train_ds[i]
        reference = row["tagged_text"]
        audio_array = row["audio"]["array"]

        # Build inputs
        inputs = processor.apply_transcription_request(
            language="en",
            audio=[audio_array],
            format=["WAV"],
            model_id=MODEL_ID,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate N completions with sampling
        best_reward = -1.0
        best_prediction = None
        best_details = None

        for s in range(num_samples):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )

            input_len = inputs["input_ids"].shape[1]
            prediction = processor.tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )

            reward, details = compute_reward(prediction, reference)
            if reward > best_reward:
                best_reward = reward
                best_prediction = prediction
                best_details = details

        curated_data.append({
            "audio_idx": i,
            "reference": reference,
            "best_prediction": best_prediction,
            "reward": best_reward,
            **best_details,
        })
        total_reward += best_reward

        if i < 5 or i % 100 == 0:
            print(f"  [{i}/{len(train_ds)}] reward={best_reward:.3f} "
                  f"wer_acc={best_details['wer_accuracy']:.3f} "
                  f"tag_f1={best_details['tag_f1']:.3f} "
                  f"hall={best_details['hall_rate']:.3f}")
            if i < 3:
                print(f"    ref: {reference[:80]}...")
                print(f"    best: {best_prediction[:80]}...")

    avg_reward = total_reward / len(curated_data)
    print(f"\nGeneration complete! Avg reward: {avg_reward:.4f}")
    print(f"Curated {len(curated_data)} samples")

    # Save curated data
    output_path = "/output/rl_curated_data.json"
    with open(output_path, "w") as f:
        json.dump(curated_data, f)
    output_vol.commit()
    print(f"Saved curated data to {output_path}")

    return {"avg_reward": avg_reward, "num_samples": len(curated_data)}


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
def rl_finetune(learning_rate: float = 5e-5, num_epochs: int = 1, push_to_hub: bool = True):
    """Step 2: SFT on curated best completions (RAFT)."""
    import torch
    import wandb
    import json
    from pathlib import Path
    from datasets import Dataset, Audio, load_from_disk
    from transformers import (
        VoxtralForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )
    from peft import PeftModel

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load curated data ---
    with open("/output/rl_curated_data.json") as f:
        curated_data = json.load(f)
    print(f"Loaded {len(curated_data)} curated samples")

    # Filter out low-reward samples (bottom 10%)
    rewards = [d["reward"] for d in curated_data]
    threshold = sorted(rewards)[len(rewards) // 10]
    curated_data = [d for d in curated_data if d["reward"] > threshold]
    print(f"After filtering (reward > {threshold:.3f}): {len(curated_data)} samples")

    # --- Load original audio dataset ---
    ds = load_from_disk("/data/processed")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    train_ds = ds["train"]

    # Build RL training dataset: use original audio + best prediction as target
    rl_examples = []
    for d in curated_data:
        idx = d["audio_idx"]
        row = train_ds[idx]
        rl_examples.append({
            "audio": row["audio"],
            "tagged_text": d["best_prediction"],  # RL target = best sampled completion
        })

    rl_dataset = Dataset.from_list(rl_examples)
    rl_dataset = rl_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"RL training dataset: {len(rl_dataset)} samples")

    # --- W&B ---
    run = wandb.init(
        project="evoxtral",
        name=f"rl-raft-lr{learning_rate}-ep{num_epochs}",
        config={
            "method": "RAFT (rejection sampling + SFT)",
            "base_adapter": "evoxtral-lora (SFT)",
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "num_curated": len(rl_dataset),
            "reward_threshold": threshold,
        },
        tags=["evoxtral", "rl", "raft", "rejection-sampling"],
    )

    avg_reward = sum(d["reward"] for d in curated_data) / len(curated_data)
    wandb.log({"rl/curated_samples": len(curated_data), "rl/avg_reward": avg_reward})

    # --- Load SFT model ---
    print("Loading SFT model for RL finetuning...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)
    # Unfreeze LoRA for continued training
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    model.print_trainable_parameters()

    # --- Data Collator (same as SFT) ---
    class VoxtralDataCollator:
        def __init__(self, processor, model_id, max_text_len=512):
            self.processor = processor
            self.model_id = model_id
            self.max_text_len = max_text_len
            self.pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

        def __call__(self, examples):
            texts = [ex["tagged_text"] for ex in examples]
            audios = [ex["audio"]["array"] for ex in examples]

            prompt = self.processor.apply_transcription_request(
                language="en",
                model_id=self.model_id,
                audio=audios,
                format=["WAV"] * len(audios),
                return_tensors="pt",
            )
            passthrough = {k: v for k, v in prompt.items()
                          if k not in ("input_ids", "attention_mask")}

            prompt_ids = prompt["input_ids"]
            prompt_attn = prompt["attention_mask"]
            B = prompt_ids.size(0)

            tok = self.processor.tokenizer
            text_tok = tok(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors=None,
            )
            text_ids_list = text_tok["input_ids"]

            input_ids, attention_mask, labels = [], [], []
            for i in range(B):
                p_ids = prompt_ids[i].tolist()
                p_att = prompt_attn[i].tolist()
                t_ids = text_ids_list[i]

                ids = p_ids + t_ids + [tok.eos_token_id]
                attn = p_att + [1] * (len(t_ids) + 1)
                lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

                input_ids.append(ids)
                attention_mask.append(attn)
                labels.append(lab)

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
            for k, v in passthrough.items():
                batch[k] = v
            return batch

    collator = VoxtralDataCollator(processor, MODEL_ID)

    # --- Training args (lower LR, 1 epoch) ---
    training_args = TrainingArguments(
        output_dir=RL_OUTPUT,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=rl_dataset,
        data_collator=collator,
    )

    print("Starting RL finetuning...")
    train_result = trainer.train()

    wandb.log({
        "rl/final_loss": train_result.metrics.get("train_loss", 0),
        "rl/runtime_seconds": train_result.metrics.get("train_runtime", 0),
    })

    # Save adapter
    print(f"Saving RL adapter to {RL_OUTPUT}")
    trainer.save_model(RL_OUTPUT)
    processor.save_pretrained(RL_OUTPUT)

    # Log as W&B artifact
    artifact = wandb.Artifact(
        "evoxtral-rl-adapter",
        type="model",
        metadata={"method": "RAFT", "base": "evoxtral-lora"},
    )
    artifact.add_dir(RL_OUTPUT)
    run.log_artifact(artifact)

    # Push to Hub
    if push_to_hub:
        from huggingface_hub import HfApi
        HUB_ID = "YongkangZOU/evoxtral-rl"
        print(f"Pushing to HuggingFace Hub: {HUB_ID}")
        try:
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            api.create_repo(HUB_ID, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=RL_OUTPUT,
                repo_id=HUB_ID,
                repo_type="model",
                commit_message=f"RL adapter (RAFT): lr={learning_rate}, ep={num_epochs}",
            )
            print(f"Pushed to {HUB_ID}")
        except Exception as e:
            print(f"Hub push failed: {e}")

    output_vol.commit()
    wandb.finish()
    print("RL finetuning complete!")
    return train_result.metrics


@app.local_entrypoint()
def main(
    num_samples: int = 4,
    temperature: float = 0.7,
    lr: float = 5e-5,
    epochs: int = 1,
    push_to_hub: bool = True,
    finetune_only: bool = False,
):
    if not finetune_only:
        print("Step 1: Generating and scoring completions...")
        gen_results = generate_and_score.remote(
            num_samples=num_samples,
            temperature=temperature,
        )
        print(f"Generation results: {gen_results}")

    print("\nStep 2: RL finetuning on curated data...")
    ft_results = rl_finetune.remote(
        learning_rate=lr,
        num_epochs=epochs,
        push_to_hub=push_to_hub,
    )
    print(f"RL finetune results: {ft_results}")
    print("\nDone! Run eval with: modal run scripts/train_modal.py --eval-only")
    print("(Update adapter_path in evaluate() to /output/evoxtral-rl)")
