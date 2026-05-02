"""
Geometric audit — orthogonality analysis of per-task LoRA adapters.

Usage (after run.py has saved <output_dir>/<method>_adapters.pt):
    python scripts/run_analysis.py --adapters results/olora_adapters.pt \
                                   --method   olora \
                                   --tasks    ag_news amazon_polarity dbpedia_14 yahoo_answers_topics \
                                   --rank     8 \
                                   --output   results/analysis
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.weight_extraction import load_adapters
from src.analysis.orthogonality import (
    task_similarity_matrix,
    plot_similarity_heatmap,
    summary_stats,
)
from src.analysis.per_layer_analysis import (
    per_layer_similarity,
    average_off_diagonal_per_layer,
    plot_layer_orthogonality,
    plot_module_orthogonality,
    print_summary,
)


def run_analysis(adapters: dict, method_name: str, task_names: list,
                 output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    num_tasks = len(adapters)

    # ── 1 & 2: Task-pair similarity matrices (A-level and AB-product) ─────────
    sim_A  = task_similarity_matrix(adapters, mode='A')
    sim_AB = task_similarity_matrix(adapters, mode='AB')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f'{method_name} — task-pair cosine similarity')
    plot_similarity_heatmap(sim_A,  f'A matrices',   task_names, ax=axes[0])
    plot_similarity_heatmap(sim_AB, f'B@A product',  task_names, ax=axes[1])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{method_name}_task_similarity.png'), dpi=150)
    plt.close(fig)
    print(f"Saved task-pair similarity heatmaps.")

    # ── 3: Layer-wise orthogonality plot ──────────────────────────────────────
    pl_A  = per_layer_similarity(adapters, mode='A')
    pl_AB = per_layer_similarity(adapters, mode='AB')

    scores_A  = average_off_diagonal_per_layer(pl_A)
    scores_AB = average_off_diagonal_per_layer(pl_AB)

    fig = plot_layer_orthogonality(scores_A, scores_AB, method_name=method_name)
    fig.savefig(os.path.join(output_dir, f'{method_name}_layer_orthogonality.png'), dpi=150)
    plt.close(fig)
    print(f"Saved layer-wise orthogonality plot.")

    # ── 4: Per-module orthogonality heatmaps ──────────────────────────────────
    fig = plot_module_orthogonality(scores_A, scores_AB, method_name=method_name)
    fig.savefig(os.path.join(output_dir, f'{method_name}_module_orthogonality.png'), dpi=150)
    plt.close(fig)
    print(f"Saved per-module orthogonality plot.")

    # ── 5: Summary statistics ─────────────────────────────────────────────────
    stats_A  = summary_stats(sim_A)
    stats_AB = summary_stats(sim_AB)
    print_summary(method_name, stats_A, stats_AB)

    results = {
        'method':           method_name,
        'task_names':       task_names,
        'sim_matrix_A':     sim_A.tolist(),
        'sim_matrix_AB':    sim_AB.tolist(),
        'summary_A':        stats_A,
        'summary_AB':       stats_AB,
        'layer_scores_A':   {k: float(v) for k, v in scores_A.items()},
        'layer_scores_AB':  {k: float(v) for k, v in scores_AB.items()},
    }
    out_path = os.path.join(output_dir, f'{method_name}_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary JSON to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', required=True, help='Path to <method>_adapters.pt')
    parser.add_argument('--method',   required=True, help='Method name (olora / inclora)')
    parser.add_argument('--tasks',    nargs='+',
                        default=['ag_news', 'amazon_polarity', 'dbpedia_14', 'yahoo_answers_topics'])
    parser.add_argument('--rank',     type=int, default=8)
    parser.add_argument('--output',   default='results/analysis')
    args = parser.parse_args()

    adapters = load_adapters(args.adapters)
    run_analysis(adapters, args.method, args.tasks, args.output)


if __name__ == '__main__':
    main()
