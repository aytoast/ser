"""Evoxtral LoRA finetuning script for Voxtral-Mini-3B.

Finetunes Voxtral to produce transcriptions with inline ElevenLabs v3 audio tags.
Uses LoRA + NEFTune + W&B tracking + artifact logging.
"""

import os
import torch
import wandb
from pathlib import Path
from datasets import load_from_disk, Audio
from transformers import (
    VoxtralForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv

from .config import EvoxtralTrainingConfig

load_dotenv()

TRANSCRIPTION_PROMPT = "Transcribe this audio with expressive tags."


def load_dataset(config: EvoxtralTrainingConfig):
    """Load the processed HuggingFace dataset from disk."""
    ds = load_from_disk(config.dataset_path)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"Dataset loaded: train={len(ds['train'])}, val={len(ds['validation'])}, test={len(ds['test'])}")
    return ds


def log_dataset_artifact(config: EvoxtralTrainingConfig, run: wandb.sdk.wandb_run.Run):
    """Log the training dataset as a W&B artifact for reproducibility."""
    artifact = wandb.Artifact(
        name=config.wandb_dataset_artifact_name,
        type="dataset",
        metadata={
            "dataset_path": config.dataset_path,
            "project": "evoxtral",
            "description": "ElevenLabs v3 tagged speech dataset (audio + tagged transcription pairs)",
        },
    )
    artifact.add_file("data/scripts/scripts.json", name="scripts.json")
    artifact.add_file("data/audio/manifest.json", name="manifest.json")
    run.log_artifact(artifact)
    print(f"Dataset artifact logged: {config.wandb_dataset_artifact_name}")


def setup_model(config: EvoxtralTrainingConfig):
    """Load Voxtral model with LoRA adapter."""
    dtype = getattr(torch, config.torch_dtype)

    processor = AutoProcessor.from_pretrained(config.model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Freeze the Whisper encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


class VoxtralDataCollator:
    """Collator that builds (audio, prompt) -> tagged_text training pairs."""

    def __init__(self, processor, max_seq_length: int = 2048):
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, examples: list[dict]) -> dict:
        conversations = []
        target_texts = []

        for ex in examples:
            audio = ex["audio"]
            tagged_text = ex["tagged_text"]

            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio["array"]},
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": tagged_text,
                },
            ]
            conversations.append(conv)
            target_texts.append(tagged_text)

        # Process all conversations
        inputs = self.processor.apply_chat_template(
            conversations,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Build labels: copy input_ids, mask everything except assistant response
        labels = inputs["input_ids"].clone()

        # Mask padding tokens
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask all prompt/user tokens — only train on assistant response
        for i, target in enumerate(target_texts):
            target_ids = self.processor.tokenizer.encode(target, add_special_tokens=False)
            seq_len = (labels[i] != -100).sum().item()

            target_len = len(target_ids)
            prompt_len = seq_len - target_len
            if prompt_len > 0:
                labels[i, :prompt_len] = -100

        inputs["labels"] = labels
        return inputs


def log_adapter_artifact(config: EvoxtralTrainingConfig, run: wandb.sdk.wandb_run.Run):
    """Log the trained LoRA adapter as a W&B artifact and link to registry."""
    artifact = wandb.Artifact(
        name=config.wandb_artifact_name,
        type="model",
        metadata={
            "wandb.base_model": config.model_id,
            "lora_rank": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "neftune_noise_alpha": config.neftune_noise_alpha,
            "learning_rate": config.learning_rate,
            "epochs": config.num_train_epochs,
            "hub_model_id": config.hub_model_id,
        },
    )
    artifact.add_dir(config.output_dir)
    logged = run.log_artifact(artifact)

    # Link to W&B model registry
    run.link_artifact(
        artifact=logged,
        target_path="wandb-registry-model/evoxtral-lora",
    )
    print(f"LoRA adapter artifact logged and linked to registry: {config.wandb_artifact_name}")


def train(config: EvoxtralTrainingConfig | None = None):
    """Run the full finetuning pipeline."""
    if config is None:
        config = EvoxtralTrainingConfig()

    # Enable auto model logging at end of training
    os.environ["WANDB_LOG_MODEL"] = config.wandb_log_model

    # Init W&B
    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "model_id": config.model_id,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_target_modules": config.lora_target_modules,
            "neftune_noise_alpha": config.neftune_noise_alpha,
            "learning_rate": config.learning_rate,
            "lr_scheduler_type": config.lr_scheduler_type,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "epochs": config.num_train_epochs,
            "batch_size": config.per_device_train_batch_size,
            "grad_accum": config.gradient_accumulation_steps,
            "effective_batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps,
            "max_seq_length": config.max_seq_length,
            "bf16": config.bf16,
            "gradient_checkpointing": config.gradient_checkpointing,
        },
        tags=["evoxtral", "voxtral", "lora", "sft", "neftune"],
    )

    # Log dataset artifact
    log_dataset_artifact(config, run)

    # Load data
    ds = load_dataset(config)

    # Log dataset stats
    wandb.log({
        "dataset/train_size": len(ds["train"]),
        "dataset/val_size": len(ds["validation"]),
        "dataset/test_size": len(ds["test"]),
    })

    # Load model + LoRA
    model, processor = setup_model(config)

    # Log trainable params info
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    wandb.log({
        "model/total_params": total,
        "model/trainable_params": trainable,
        "model/trainable_pct": trainable / total * 100,
    })

    # Data collator
    collator = VoxtralDataCollator(processor, config.max_seq_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=config.neftune_noise_alpha,
        # Logging & saving
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # W&B
        report_to="wandb",
        run_name=config.wandb_run_name,
        # Misc
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        # Hub
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id if config.push_to_hub else None,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Log final metrics
    wandb.log({
        "train/final_loss": train_result.metrics.get("train_loss", 0),
        "train/total_steps": train_result.metrics.get("train_steps", 0),
        "train/runtime_seconds": train_result.metrics.get("train_runtime", 0),
    })

    # Save adapter locally
    print(f"Saving adapter to {config.output_dir}")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)

    # Log LoRA adapter as W&B artifact + link to registry
    log_adapter_artifact(config, run)

    # Push to HuggingFace Hub
    if config.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {config.hub_model_id}")
        trainer.push_to_hub()

    wandb.finish()
    print("Training complete!")

    return train_result


if __name__ == "__main__":
    train()
