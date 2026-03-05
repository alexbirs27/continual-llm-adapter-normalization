from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class LoraConfig:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs_per_task: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 256
    lambda_1: float = 0.0
    lambda_2: float = 0.0
    max_samples_per_task: int = 20000
    task_order: List[str] = field(
        default_factory=lambda: [
            "ag_news",
            "yelp_review_full",
            "amazon_polarity",
            "dbpedia_14",
            "yahoo_answers_topics",
        ]
    )


def load_config(path: str):
    """Load config from YAML file, return (method, model_name, LoraConfig, TrainingConfig)."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    method = raw["method"]
    model_name = raw["model_name"]

    lora_raw = raw.get("lora", {})
    lora_cfg = LoraConfig(**lora_raw)

    training_raw = raw.get("training", {})
    # Ensure numeric types are correct (YAML may parse scientific notation as str)
    for key in ("lr", "lambda_1", "lambda_2", "dropout"):
        if key in training_raw:
            training_raw[key] = float(training_raw[key])
        if key in lora_raw:
            lora_raw[key] = float(lora_raw[key])
    training_cfg = TrainingConfig(**training_raw)

    return method, model_name, lora_cfg, training_cfg
