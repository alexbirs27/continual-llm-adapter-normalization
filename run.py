"""Main entry point for continual learning experiments."""

import argparse
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.datasets import load_all_datasets
from src.methods.inclora import IncLoRA
from src.methods.olora import OLoRA
from src.methods.ella import ELLA
from src.training.continual_trainer import ContinualTrainer
from src.utils.config import load_config

ALL_CONFIGS = [
    "configs/inclora.yaml",
    "configs/olora.yaml",
    "configs/ella.yaml",
]


def run_single(config_path, output_dir="results"):
    """Run a single method from a config file."""
    method_name, model_name, lora_config, training_config = load_config(config_path)

    print(f"\n{'#'*60}")
    print(f"# Method: {method_name}")
    print(f"# Model: {model_name}")
    print(f"# Tasks: {training_config.task_order}")
    print(f"# Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'#'*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map=device,
    )

    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()

    # Load datasets
    datasets = load_all_datasets(
        training_config.task_order,
        tokenizer,
        max_length=training_config.max_seq_length,
        max_samples=training_config.max_samples_per_task,
    )

    # Instantiate method
    if method_name == "inclora":
        method = IncLoRA(model, lora_config)
    elif method_name == "olora":
        method = OLoRA(
            model,
            lora_config,
            lambda_1=training_config.lambda_1,
            lambda_2=training_config.lambda_2,
        )
    elif method_name == "ella":
        method = ELLA(
            model,
            lora_config,
            lambd=training_config.lambda_1,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Train
    trainer = ContinualTrainer(method, datasets, training_config, tokenizer)
    results_matrix, metrics = trainer.train_all_tasks()

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "method": method_name,
        "model": model_name,
        "task_order": training_config.task_order,
        "results_matrix": results_matrix,
        "metrics": metrics,
    }
    output_path = os.path.join(output_dir, f"{method_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return method_name, metrics


def main():
    parser = argparse.ArgumentParser(description="Continual Learning with LoRA")
    parser.add_argument("--config", type=str, help="Path to a single config YAML")
    parser.add_argument("--all", action="store_true", help="Run all methods (inclora, olora, ella)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    if not args.all and not args.config:
        parser.error("Provide --config <path> or --all")

    if args.config:
        run_single(args.config, args.output_dir)
    elif args.all:
        all_metrics = {}
        for config_path in ALL_CONFIGS:
            name, metrics = run_single(config_path, args.output_dir)
            all_metrics[name] = metrics

            # Free GPU memory between runs
            torch.cuda.empty_cache()
            gc.collect()

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON ACROSS METHODS")
        print(f"{'='*60}")
        print(f"{'Method':<15} {'ACC':>10} {'BWT':>10}")
        print("-" * 35)
        for name, m in all_metrics.items():
            print(f"{name:<15} {m['acc']:>10.4f} {m['bwt']:>10.4f}")


if __name__ == "__main__":
    main()
