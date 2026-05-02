"""Per-layer and per-module orthogonality analysis of LoRA adapters."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from src.analysis.orthogonality import cosine_sim, mean_off_diagonal
from src.analysis.weight_extraction import delta_W, layer_index, module_type


def _get_tensor(entry: dict, mode: str):
    return entry['A'] if mode == 'A' else delta_W(entry)


def per_layer_similarity(adapters: dict, mode: str = 'A') -> dict:
    """
    Compute a (n_tasks × n_tasks) cosine similarity matrix per layer.

    Returns:
        {layer_name: np.ndarray of shape (n_tasks, n_tasks)}
    """
    num_tasks   = len(adapters)
    layer_names = list(adapters[0].keys())
    result = {}

    for layer in layer_names:
        mat = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                if layer in adapters[i] and layer in adapters[j]:
                    m_i = _get_tensor(adapters[i][layer], mode)
                    m_j = _get_tensor(adapters[j][layer], mode)
                    mat[i, j] = cosine_sim(m_i, m_j)
        result[layer] = mat

    return result


def average_off_diagonal_per_layer(per_layer_mats: dict) -> dict:
    """
    Reduce each layer's (n_tasks × n_tasks) matrix to one scalar:
    the mean off-diagonal cosine similarity (inter-task interference).

    Returns:
        {layer_name: float}
    """
    return {name: mean_off_diagonal(mat) for name, mat in per_layer_mats.items()}


def _sorted_layers(scores: dict):
    """Sort layers by index for plotting."""
    return sorted(scores.items(), key=lambda kv: layer_index(kv[0]))


def plot_layer_orthogonality(per_layer_scores_A: dict, per_layer_scores_AB: dict,
                             method_name: str = '', ax=None):
    """
    Line plot: mean off-diagonal cosine similarity vs layer index.
    Shows both A-level and AB-product level on the same axes.
    Lower = more orthogonal (better).
    """
    layers_A  = _sorted_layers(per_layer_scores_A)
    layers_AB = _sorted_layers(per_layer_scores_AB)

    idxs_A  = [layer_index(k) for k, _ in layers_A]
    vals_A  = [v for _, v in layers_A]
    idxs_AB = [layer_index(k) for k, _ in layers_AB]
    vals_AB = [v for _, v in layers_AB]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(idxs_A,  vals_A,  marker='o', label='A matrices',  linewidth=1.5)
    ax.plot(idxs_AB, vals_AB, marker='s', label='B@A product', linewidth=1.5, linestyle='--')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8, label='Perfect orthogonality')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean inter-task cosine similarity')
    ax.set_title(f'Layer-wise orthogonality — {method_name}')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    return fig


def per_module_scores(per_layer_scores: dict) -> dict:
    """
    Group mean off-diagonal scores by module type (q_proj / v_proj).

    Returns:
        {'q_proj': [(layer_idx, score), ...], 'v_proj': [...]}
    """
    groups = defaultdict(list)
    for layer_name, score in per_layer_scores.items():
        mod = module_type(layer_name)
        idx = layer_index(layer_name)
        groups[mod].append((idx, score))

    for mod in groups:
        groups[mod].sort(key=lambda x: x[0])

    return dict(groups)


def plot_module_orthogonality(per_layer_scores_A: dict, per_layer_scores_AB: dict,
                              method_name: str = ''):
    """
    Two-panel plot — one panel per module type (q_proj, v_proj).
    Each panel shows A-level and AB-product similarity vs layer index.
    """
    modules = ['q_proj', 'v_proj']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(f'Per-module orthogonality — {method_name}')

    for ax, mod in zip(axes, modules):
        groups_A  = per_module_scores({k: v for k, v in per_layer_scores_A.items()  if mod in k})
        groups_AB = per_module_scores({k: v for k, v in per_layer_scores_AB.items() if mod in k})

        pts_A  = groups_A.get(mod, [])
        pts_AB = groups_AB.get(mod, [])

        if pts_A:
            idxs, vals = zip(*pts_A)
            ax.plot(idxs, vals, marker='o', label='A matrices', linewidth=1.5)
        if pts_AB:
            idxs, vals = zip(*pts_AB)
            ax.plot(idxs, vals, marker='s', label='B@A product', linewidth=1.5, linestyle='--')

        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax.set_title(mod)
        ax.set_xlabel('Layer index')
        ax.set_ylabel('Mean inter-task cosine similarity')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def print_summary(method_name: str, stats_A: dict, stats_AB: dict):
    print(f"\n{'='*55}")
    print(f"  {method_name} — orthogonality summary")
    print(f"{'='*55}")
    print(f"  {'Metric':<18} {'A matrices':>14} {'B@A product':>14}")
    print(f"  {'-'*46}")
    for key in ('mean', 'std', 'min', 'max'):
        print(f"  {key:<18} {stats_A[key]:>14.4f} {stats_AB[key]:>14.4f}")
