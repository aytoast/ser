"""Training configuration for Evoxtral LoRA finetuning."""

from dataclasses import dataclass, field


@dataclass
class EvoxtralTrainingConfig:
    # Model
    model_id: str = "mistralai/Voxtral-Mini-3B-2507"
    torch_dtype: str = "bfloat16"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "multi_modal_projector.linear_1",
        "multi_modal_projector.linear_2",
    ])

    # NEFTune
    neftune_noise_alpha: float = 5.0

    # Training hyperparams
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Precision & memory
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Data
    dataset_path: str = "data/processed"
    max_seq_length: int = 2048

    # Output
    output_dir: str = "model/evoxtral-lora"
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3

    # W&B
    wandb_project: str = "evoxtral"
    wandb_run_name: str | None = None
    wandb_log_model: str = "end"  # auto-log model artifact at end of training
    wandb_artifact_name: str = "evoxtral-lora-adapter"
    wandb_dataset_artifact_name: str = "evoxtral-dataset"

    # HuggingFace Hub
    push_to_hub: bool = True
    hub_model_id: str = "mistral-hackaton-2026/evoxtral-lora"
