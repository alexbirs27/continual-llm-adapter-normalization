"""Per-layer and per-module orthogonality analysis of LoRA adapters."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from src.analysis.orthogonality import METRICS, METRIC_LABELS, mean_off_diagonal
from src.analysis.weight_extraction import delta_W, layer_index, module_type


def _get_tensor(entry: dict, mode: str):
    return entry['A'] if mode == 'A' else delta_W(entry)


def per_layer_similarity(adapters: dict, mode: str = 'A',
                         metric: str = 'cosine') -> dict:
    """Compute a (n_tasks × n_tasks) similarity matrix per layer.

    Returns:
        {layer_name: np.ndarray of shape (n_tasks, n_tasks)}
    """
    num_tasks   = len(adapters)
    layer_names = list(adapters[0].keys())
    sim_fn      = METRICS[metric]
    result      = {}

    for layer in layer_names:
        mat = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                if layer in adapters[i] and layer in adapters[j]:
                    m_i = _get_tensor(adapters[i][layer], mode)
                    m_j = _get_tensor(adapters[j][layer], mode)
                    mat[i, j] = sim_fn(m_i, m_j)
        result[layer] = mat

    return result


def average_off_diagonal_per_layer(per_layer_mats: dict) -> dict:
    """Reduce each layer's (n_tasks × n_tasks) matrix to one scalar.

    Returns:
        {layer_name: float}
    """
    return {name: mean_off_diagonal(mat) for name, mat in per_layer_mats.items()}


def _sorted_layers(scores: dict):
    return sorted(scores.items(), key=lambda kv: layer_index(kv[0]))


def plot_layer_orthogonality(scores_per_metric: dict, method_name: str = ''):
    """Line plot: mean off-diagonal similarity vs layer index, one curve per metric.

    Args:
        scores_per_metric: {metric_name: {layer_name: float}}
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for metric, scores in scores_per_metric.items():
        layers = _sorted_layers(scores)
        idxs = [layer_index(k) for k, _ in layers]
        vals = [v for _, v in layers]
        ax.plot(idxs, vals, marker='o', label=METRIC_LABELS[metric], linewidth=1.5)

    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8, label='Perfect orthogonality')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean inter-task similarity')
    ax.set_title(f'Layer-wise orthogonality (A matrices) — {method_name}')
    ax.legend(fontsize=8)
    return fig


def per_module_scores(per_layer_scores: dict) -> dict:
    """Group mean off-diagonal scores by module type (q_proj / v_proj)."""
    groups = defaultdict(list)
    for layer_name, score in per_layer_scores.items():
        mod = module_type(layer_name)
        idx = layer_index(layer_name)
        groups[mod].append((idx, score))
    for mod in groups:
        groups[mod].sort(key=lambda x: x[0])
    return dict(groups)


def plot_module_orthogonality(scores_per_metric: dict, method_name: str = ''):
    """Two-panel plot (q_proj / v_proj), one curve per metric per panel.

    Args:
        scores_per_metric: {metric_name: {layer_name: float}}
    """
    modules = ['q_proj', 'v_proj']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(f'Per-module orthogonality (A matrices) — {method_name}')

    for ax, mod in zip(axes, modules):
        for metric, scores in scores_per_metric.items():
            filtered = {k: v for k, v in scores.items() if mod in k}
            groups = per_module_scores(filtered)
            pts = groups.get(mod, [])
            if pts:
                idxs, vals = zip(*pts)
                ax.plot(idxs, vals, marker='o', label=METRIC_LABELS[metric], linewidth=1.5)

        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax.set_title(mod)
        ax.set_xlabel('Layer index')
        ax.set_ylabel('Mean inter-task similarity')
        ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_ab_layer_orthogonality(scores_A: dict, scores_AB: dict, method_name: str = ''):
    """Line plot comparing A-level vs AB-product cosine similarity per layer."""
    layers_A  = _sorted_layers(scores_A)
    layers_AB = _sorted_layers(scores_AB)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([layer_index(k) for k, _ in layers_A],  [v for _, v in layers_A],
            marker='o', label='A matrices (cosine)', linewidth=1.5)
    ax.plot([layer_index(k) for k, _ in layers_AB], [v for _, v in layers_AB],
            marker='s', label='B@A product (cosine)', linewidth=1.5, linestyle='--')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8, label='Perfect orthogonality')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean inter-task cosine similarity')
    ax.set_title(f'Layer-wise orthogonality — {method_name}')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    return fig


def print_summary(method_name: str, stats_per_metric: dict, stats_AB: dict):
    print(f"\n{'='*65}")
    print(f"  {method_name} — orthogonality summary (A matrices)")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'-'*62}")
    for metric, stats in stats_per_metric.items():
        print(f"  {metric.capitalize():<22} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
              f"{stats['min']:>10.4f} {stats['max']:>10.4f}")
    print(f"  {'B@A cosine':<22} {stats_AB['mean']:>10.4f} {stats_AB['std']:>10.4f} "
          f"{stats_AB['min']:>10.4f} {stats_AB['max']:>10.4f}")
