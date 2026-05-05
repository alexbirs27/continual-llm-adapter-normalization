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

    Averages across all module types at each layer index so there is exactly
    one data point per transformer block.

    Args:
        scores_per_metric: {metric_name: {layer_name: float}}
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for metric, scores in scores_per_metric.items():
        layer_vals = defaultdict(list)
        for k, v in scores.items():
            layer_vals[layer_index(k)].append(v)
        idxs = sorted(layer_vals.keys())
        vals = [float(np.mean(layer_vals[idx])) for idx in idxs]
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
    """Grid plot with one panel per module type, one curve per metric per panel.

    Module types are discovered dynamically from the layer names in scores,
    so the plot works for any combination of target_modules.

    Args:
        scores_per_metric: {metric_name: {layer_name: float}}
    """
    first_scores = next(iter(scores_per_metric.values()))
    modules = sorted(
        {module_type(k) for k in first_scores} - {'unknown'},
        key=lambda m: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].index(m)
        if m in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] else 99
    )

    n = len(modules)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True, squeeze=False)
    fig.suptitle(f'Per-module orthogonality (A matrices) — {method_name}')

    for idx, mod in enumerate(modules):
        ax = axes[idx // ncols][idx % ncols]
        for metric, scores in scores_per_metric.items():
            filtered = {k: v for k, v in scores.items() if module_type(k) == mod}
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

    # Hide any unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    return fig


def plot_ab_layer_orthogonality(scores_A_cosine: dict, scores_AB: dict, method_name: str = ''):
    """Line plot comparing A cosine vs B@A product metrics per layer.

    Args:
        scores_A_cosine: {layer_name: float} — A matrix cosine similarity
        scores_AB:       {metric_name: {layer_name: float}} — B@A metrics
    """
    from src.analysis.orthogonality import METRIC_LABELS

    def _avg_by_layer(scores):
        layer_vals = defaultdict(list)
        for k, v in scores.items():
            layer_vals[layer_index(k)].append(v)
        idxs = sorted(layer_vals.keys())
        return idxs, [float(np.mean(layer_vals[idx])) for idx in idxs]

    fig, ax = plt.subplots(figsize=(10, 4))

    idxs, vals = _avg_by_layer(scores_A_cosine)
    ax.plot(idxs, vals, marker='o', label='A matrices (cosine)', linewidth=1.5)

    markers = ['s', '^', 'D']
    linestyles = ['--', ':', '-.']
    for (marker, ls), (metric, scores) in zip(zip(markers, linestyles), scores_AB.items()):
        idxs, vals = _avg_by_layer(scores)
        ax.plot(idxs, vals, marker=marker, linestyle=ls,
                label=f'B@A ({METRIC_LABELS[metric]})', linewidth=1.5)

    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8, label='Perfect orthogonality')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean inter-task similarity')
    ax.set_title(f'Layer-wise orthogonality — {method_name}')
    ax.legend(fontsize=8)
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
