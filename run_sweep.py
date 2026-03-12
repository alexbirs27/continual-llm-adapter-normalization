"""Hyperparameter sweep for lambda selection (OLoRA / ELLA).

Sweeps lambda over {3e-4, 3e-5, 3e-6, 3e-7}, trains full 5-task sequence
for each, evaluates on dev split, picks best lambda by ACC.
"""

import argparse
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.datasets import load_all_datasets
from src.methods.olora import OLoRA
from src.methods.ella import ELLA
from src.training.continual_trainer import ContinualTrainer
from src.utils.config import load_config

LAMBDA_GRID = [3e-4, 3e-5, 3e-6, 3e-7]

METHOD_CONFIGS = {
    "olora": "configs/olora.yaml",
    "ella": "configs/ella.yaml",
}


def run_sweep_single(method_name, lambd, tokenizer, training_config, lora_config, model_name):
    """Run a single lambda value and return metrics."""
    print(f"\n{'#'*60}")
    print(f"# Sweep: {method_name} | lambda = {lambd}")
    print(f"{'#'*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map=device,
    )
    model.gradient_checkpointing_enable()

    datasets = load_all_datasets(
        training_config.task_order,
        tokenizer,
        max_length=training_config.max_seq_length,
        max_samples=training_config.max_samples_per_task,
    )

    if method_name == "olora":
        method = OLoRA(model, lora_config, lambda_1=lambd, lambda_2=0.0)
    elif method_name == "ella":
        method = ELLA(model, lora_config, lambd=lambd)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    trainer = ContinualTrainer(method, datasets, training_config, tokenizer, dev_mode=True)
    results_matrix, metrics = trainer.train_all_tasks()

    # Clean up
    del model, method, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return results_matrix, metrics


def main():
    parser = argparse.ArgumentParser(description="Lambda hyperparameter sweep")
    parser.add_argument("--method", type=str, required=True, choices=["olora", "ella"],
                        help="Method to sweep (olora or ella)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    config_path = METHOD_CONFIGS[args.method]
    method_name, model_name, lora_config, training_config = load_config(config_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}
    best_lambda = None
    best_acc = -1.0

    for lambd in LAMBDA_GRID:
        results_matrix, metrics = run_sweep_single(
            args.method, lambd, tokenizer, training_config, lora_config, model_name
        )
        all_results[str(lambd)] = {
            "lambda": lambd,
            "results_matrix": results_matrix,
            "metrics": metrics,
        }
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_lambda = lambd

    # Summary
    print(f"\n{'='*60}")
    print(f"SWEEP RESULTS: {args.method}")
    print(f"{'='*60}")
    print(f"{'Lambda':<15} {'ACC':>10} {'BWT':>10} {'FWT':>10}")
    print("-" * 45)
    for key, res in all_results.items():
        m = res["metrics"]
        marker = " <-- best" if res["lambda"] == best_lambda else ""
        print(f"{key:<15} {m['acc']:>10.4f} {m['bwt']:>10.4f} {m['fwt']:>10.4f}{marker}")

    print(f"\nBest lambda: {best_lambda} (ACC = {best_acc:.4f})")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "method": args.method,
        "model": model_name,
        "lambda_grid": LAMBDA_GRID,
        "best_lambda": best_lambda,
        "best_acc": best_acc,
        "sweep_results": all_results,
    }
    output_path = os.path.join(args.output_dir, f"{args.method}_sweep_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Sweep results saved to {output_path}")


if __name__ == "__main__":
    main()
