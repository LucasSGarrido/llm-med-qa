"""Training configuration for QLoRA fine-tuning on Llama models."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """LoRA adapter configuration for QLoRA fine-tuning."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Full training configuration for QLoRA fine-tuning on Kaggle."""

    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: str = "./checkpoints"
    hub_model_id: str = "SoulLucas/llama-3.2-3b-medqa"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 1024
    fp16: bool = True
    logging_steps: int = 50
    save_steps: int = 200
    eval_steps: int = 200
    load_best_model_at_end: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)


def get_bnb_config():
    """4-bit BitsAndBytes quantization config for QLoRA.

    Lazy import: torch and transformers are only needed on GPU machines.
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
