"""
Geometric audit — orthogonality analysis of per-task LoRA adapters.

Usage (after run.py has saved <output_dir>/<method>_adapters.pt):
    python scripts/run_analysis.py --adapters results/olora_adapters.pt \
                                   --method   olora \
                                   --tasks    ag_news amazon_polarity dbpedia_14 yahoo_answers_topics \
                                   --rank     8 \
                                   --output   results/analysis

Three A-matrix metrics are computed and saved:
  cosine    — cosine similarity of flattened matrices
  frobenius — normalized Frobenius norm of Gram matrix (directly corresponds to training loss)
  principal — max cosine of principal angles between row subspaces (geometric gold standard)

Plus cosine similarity on the full B@A product matrices.
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
    METRIC_LABELS,
)
from src.analysis.per_layer_analysis import (
    per_layer_similarity,
    average_off_diagonal_per_layer,
    plot_layer_orthogonality,
    plot_module_orthogonality,
    plot_ab_layer_orthogonality,
    print_summary,
)

ALL_METRICS = ['cosine', 'frobenius', 'principal']


def run_analysis(adapters: dict, method_name: str, task_names: list,
                 output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1: Task-pair similarity heatmaps for all 3 A-matrix metrics ───────────
    sim_A = {}
    for metric in ALL_METRICS:
        sim_A[metric] = task_similarity_matrix(adapters, mode='A', metric=metric)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'{method_name} — task-pair similarity (A matrices)')
    for ax, metric in zip(axes, ALL_METRICS):
        plot_similarity_heatmap(sim_A[metric], METRIC_LABELS[metric], task_names, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{method_name}_task_similarity_A.png'), dpi=150)
    plt.close(fig)
    print("Saved task-pair A-matrix similarity heatmaps.")

    # ── 2: B@A product cosine heatmap ─────────────────────────────────────────
    sim_AB = task_similarity_matrix(adapters, mode='AB', metric='cosine')
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.suptitle(f'{method_name} — task-pair cosine similarity (B@A product)')
    plot_similarity_heatmap(sim_AB, 'B@A cosine', task_names, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{method_name}_task_similarity_AB.png'), dpi=150)
    plt.close(fig)
    print("Saved task-pair B@A cosine heatmap.")

    # ── 3: Layer-wise plots — all 3 A-matrix metrics on one figure ────────────
    layer_scores_A = {}
    for metric in ALL_METRICS:
        pl = per_layer_similarity(adapters, mode='A', metric=metric)
        layer_scores_A[metric] = average_off_diagonal_per_layer(pl)

    fig = plot_layer_orthogonality(layer_scores_A, method_name=method_name)
    fig.savefig(os.path.join(output_dir, f'{method_name}_layer_orthogonality_A.png'), dpi=150)
    plt.close(fig)
    print("Saved layer-wise A-matrix orthogonality plot.")

    # ── 4: Layer-wise A vs AB (cosine only, for direct comparison) ────────────
    pl_AB = per_layer_similarity(adapters, mode='AB', metric='cosine')
    layer_scores_AB = average_off_diagonal_per_layer(pl_AB)

    fig = plot_ab_layer_orthogonality(
        layer_scores_A['cosine'], layer_scores_AB, method_name=method_name
    )
    fig.savefig(os.path.join(output_dir, f'{method_name}_layer_orthogonality_AB.png'), dpi=150)
    plt.close(fig)
    print("Saved layer-wise A vs B@A cosine plot.")

    # ── 5: Per-module heatmaps — all 3 A-matrix metrics ──────────────────────
    fig = plot_module_orthogonality(layer_scores_A, method_name=method_name)
    fig.savefig(os.path.join(output_dir, f'{method_name}_module_orthogonality.png'), dpi=150)
    plt.close(fig)
    print("Saved per-module orthogonality plot.")

    # ── 6: Summary statistics ─────────────────────────────────────────────────
    stats_A = {metric: summary_stats(sim_A[metric]) for metric in ALL_METRICS}
    stats_AB = summary_stats(sim_AB)
    print_summary(method_name, stats_A, stats_AB)

    results = {
        'method':     method_name,
        'task_names': task_names,
        'A_matrices': {
            metric: {
                'sim_matrix':   sim_A[metric].tolist(),
                'summary':      stats_A[metric],
                'layer_scores': {k: float(v) for k, v in layer_scores_A[metric].items()},
            }
            for metric in ALL_METRICS
        },
        'AB_product': {
            'sim_matrix':   sim_AB.tolist(),
            'summary':      stats_AB,
            'layer_scores': {k: float(v) for k, v in layer_scores_AB.items()},
        },
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
