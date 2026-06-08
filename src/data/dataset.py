from __future__ import annotations

from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset

DATASET_ID = "medalpaca/medical_meadow_medqa"
SYSTEM_PROMPT = (
    "You are a helpful medical assistant. "
    "Answer questions accurately based on medical knowledge."
)


def load_medalpaca(
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """Load MedAlpaca medical Q&A dataset from HuggingFace Hub."""
    ds = load_dataset(DATASET_ID, split=split)
    if max_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    return ds


def format_instruction(example: dict) -> dict:
    """Format a MedAlpaca example into Llama 3.2 Instruct chat format.

    Returns a dict with a single key ``text`` containing the fully-formatted
    conversation string ready for SFTTrainer (dataset_text_field='text').
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    user_content = instruction
    if input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"

    text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output}<|eot_id|>"
    )
    return {"text": text}


def prepare_dataset(
    max_train: int = 8000,
    max_val: int = 500,
    seed: int = 42,
) -> DatasetDict:
    """Load, format and split MedAlpaca into train/validation sets."""
    full = load_medalpaca(max_samples=max_train + max_val, seed=seed)
    split = full.train_test_split(test_size=max_val, seed=seed)
    train_ds = split["train"].map(
        format_instruction,
        remove_columns=split["train"].column_names,
    )
    val_ds = split["test"].map(
        format_instruction,
        remove_columns=split["test"].column_names,
    )
    return DatasetDict({"train": train_ds, "validation": val_ds})
