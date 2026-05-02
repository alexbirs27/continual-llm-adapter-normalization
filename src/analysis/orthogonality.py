"""Cosine similarity matrices between per-task LoRA adapter matrices."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.analysis.weight_extraction import delta_W


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    denom = a_f.norm() * b_f.norm()
    if denom < 1e-12:
        return 0.0
    return (a_f @ b_f / denom).item()


def task_similarity_matrix(adapters: dict, mode: str = 'A') -> np.ndarray:
    """
    Build an (n_tasks × n_tasks) cosine similarity matrix averaged across layers.

    Args:
        adapters: {task_id: {layer_name: {'A': Tensor, 'B': Tensor}}}
        mode:     'A'  — compare A matrices directly
                  'AB' — compare B@A low-rank update matrices

    Returns:
        sim_matrix: np.ndarray shape (n_tasks, n_tasks)
    """
    num_tasks  = len(adapters)
    layer_names = list(adapters[0].keys())
    sim_matrix  = np.zeros((num_tasks, num_tasks))

    for i in range(num_tasks):
        for j in range(num_tasks):
            sims = []
            for layer in layer_names:
                if layer not in adapters[i] or layer not in adapters[j]:
                    continue
                m_i = adapters[i][layer]['A'] if mode == 'A' else delta_W(adapters[i][layer])
                m_j = adapters[j][layer]['A'] if mode == 'A' else delta_W(adapters[j][layer])
                sims.append(cosine_sim(m_i, m_j))
            sim_matrix[i, j] = float(np.mean(sims)) if sims else 0.0

    return sim_matrix


def mean_off_diagonal(matrix: np.ndarray) -> float:
    """Average cosine similarity between distinct task pairs (i ≠ j)."""
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(matrix[mask].mean())


def plot_similarity_heatmap(sim_matrix: np.ndarray, title: str,
                            task_names=None, ax=None):
    """Annotated heatmap of a task-pair cosine similarity matrix."""
    try:
        import seaborn as sns
        use_seaborn = True
    except ImportError:
        use_seaborn = False

    n = sim_matrix.shape[0]
    labels = task_names if task_names else [f"T{i}" for i in range(n)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    if use_seaborn:
        import seaborn as sns
        sns.heatmap(
            sim_matrix, annot=True, fmt='.3f',
            cmap='RdYlGn_r', vmin=0, vmax=1,
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
    else:
        im = ax.imshow(sim_matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{sim_matrix[i, j]:.3f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax)

    mean_od = mean_off_diagonal(sim_matrix)
    ax.set_title(f"{title}\n(mean off-diag: {mean_od:.3f})")
    ax.set_xlabel("Task j")
    ax.set_ylabel("Task i")
    return fig


def summary_stats(sim_matrix: np.ndarray) -> dict:
    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = sim_matrix[mask]
    return {
        'mean':   float(off.mean()),
        'std':    float(off.std()),
        'min':    float(off.min()),
        'max':    float(off.max()),
    }
