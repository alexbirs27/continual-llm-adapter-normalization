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
AB_METRICS  = ['cosine', 'frobenius_ab']


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
    fig.savefig(os.path.join(output_dir, f'{method_name}_task_similarity_A.pdf'))
    plt.close(fig)
    print("Saved task-pair A-matrix similarity heatmaps.")

    # ── 2: B@A product heatmaps ───────────────────────────────────────────────
    sim_AB = {m: task_similarity_matrix(adapters, mode='AB', metric=m) for m in AB_METRICS}
    fig, axes = plt.subplots(1, len(AB_METRICS), figsize=(6 * len(AB_METRICS), 4))
    fig.suptitle(f'{method_name} — task-pair similarity (B@A product)')
    for ax, metric in zip(axes, AB_METRICS):
        plot_similarity_heatmap(sim_AB[metric], METRIC_LABELS[metric], task_names, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{method_name}_task_similarity_AB.pdf'))
    plt.close(fig)
    print("Saved task-pair B@A heatmaps.")

    # ── 3: Task-pair similarity heatmaps for all 3 B-matrix metrics ───────────
    sim_B = {}
    for metric in ALL_METRICS:
        sim_B[metric] = task_similarity_matrix(adapters, mode='B', metric=metric)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'{method_name} — task-pair similarity (B matrices)')
    for ax, metric in zip(axes, ALL_METRICS):
        plot_similarity_heatmap(sim_B[metric], METRIC_LABELS[metric], task_names, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{method_name}_task_similarity_B.pdf'))
    plt.close(fig)
    print("Saved task-pair B-matrix similarity heatmaps.")

    # ── 4: Layer-wise plots — all 3 A-matrix metrics on one figure ────────────
    layer_scores_A = {}
    for metric in ALL_METRICS:
        pl = per_layer_similarity(adapters, mode='A', metric=metric)
        layer_scores_A[metric] = average_off_diagonal_per_layer(pl)

    fig = plot_layer_orthogonality(layer_scores_A, method_name=method_name)
    fig.savefig(os.path.join(output_dir, f'{method_name}_layer_orthogonality_A.pdf'))
    plt.close(fig)
    print("Saved layer-wise A-matrix orthogonality plot.")

    # ── 5: Layer-wise A vs AB (cosine + frobenius_ab) ─────────────────────────
    layer_scores_AB = {}
    for metric in AB_METRICS:
        pl_AB = per_layer_similarity(adapters, mode='AB', metric=metric)
        layer_scores_AB[metric] = average_off_diagonal_per_layer(pl_AB)

    fig = plot_ab_layer_orthogonality(
        layer_scores_A['cosine'], layer_scores_AB, method_name=method_name
    )
    fig.savefig(os.path.join(output_dir, f'{method_name}_layer_orthogonality_AB.pdf'))
    plt.close(fig)
    print("Saved layer-wise A vs B@A cosine plot.")

    # ── 6: Per-module heatmaps — all 3 A-matrix metrics ──────────────────────
    fig = plot_module_orthogonality(layer_scores_A, method_name=method_name)
    fig.savefig(os.path.join(output_dir, f'{method_name}_module_orthogonality.pdf'))
    plt.close(fig)
    print("Saved per-module orthogonality plot.")

    # ── 7: Summary statistics ─────────────────────────────────────────────────
    stats_A  = {metric: summary_stats(sim_A[metric]) for metric in ALL_METRICS}
    stats_B  = {metric: summary_stats(sim_B[metric]) for metric in ALL_METRICS}
    stats_AB = {metric: summary_stats(sim_AB[metric]) for metric in AB_METRICS}
    print_summary(method_name, stats_A, stats_AB['cosine'])

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
        'B_matrices': {
            metric: {
                'sim_matrix': sim_B[metric].tolist(),
                'summary':    stats_B[metric],
            }
            for metric in ALL_METRICS
        },
        'AB_product': {
            metric: {
                'sim_matrix':   sim_AB[metric].tolist(),
                'summary':      stats_AB[metric],
                'layer_scores': {k: float(v) for k, v in layer_scores_AB[metric].items()},
            }
            for metric in AB_METRICS
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
