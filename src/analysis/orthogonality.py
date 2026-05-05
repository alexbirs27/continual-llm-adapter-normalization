"""Similarity / orthogonality metrics between per-task LoRA adapter matrices."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.analysis.weight_extraction import delta_W


# ── Pairwise similarity functions ─────────────────────────────────────────────

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two matrices treated as flat vectors.

    Captures tr(A_i A_t^T) / (||A_i||_F ||A_t||_F).
    In [-1, 1]; 0 = orthogonal as vectors.
    """
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    denom = a_f.norm() * b_f.norm()
    if denom < 1e-12:
        return 0.0
    return (a_f @ b_f / denom).item()


def gram_frobenius_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frobenius norm of the Gram matrix for B@A product matrices.

    For ΔW = B@A of shape (fan_out, fan_in), G = ΔW_i @ ΔW_j.T has shape
    (fan_out, fan_out). Returns the raw ||G||_F without normalization.
    In [0, inf); 0 = perfectly orthogonal weight updates.
    """
    G = a.float() @ b.float().T
    return torch.norm(G, p='fro').item()


def gram_frobenius(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized Frobenius norm of the Gram matrix G = a @ b.T.

    For A matrices of shape (r, fan_in), G has shape (r, r).
    Dividing by r makes it scale-invariant across ranks.
    In [0, inf); 0 = perfectly orthogonal row spaces.
    Directly corresponds to what the orthogonal loss minimizes (||G||_F^2).
    """
    r = min(a.shape)   # rank dimension, works for both (r×fan_in) and (fan_out×r)
    G = a.float() @ b.float().T
    return (torch.norm(G, p='fro') / r).item()


def max_principal_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine of the smallest principal angle between the low-rank subspaces of a and b.

    Works for both A matrices (r × fan_in) and B matrices (fan_out × r).
    The subspace is always taken along the large dimension (the one >= r):
      - A (r × fan_in): row subspace in R^{fan_in} — uses QR(a.T)
      - B (fan_out × r): column subspace in R^{fan_out} — uses QR(b)
    In [0, 1]; 0 = fully orthogonal subspaces, 1 = subspaces share a direction.
    """
    a_f = a.float()
    b_f = b.float()
    if a_f.norm() < 1e-12 or b_f.norm() < 1e-12:
        return 0.0
    # Orient so the tall dimension (>= r) is the rows of the matrix passed to QR,
    # giving Q of shape (tall_dim, r) — r orthonormal columns spanning the subspace.
    def _basis(m):
        if m.shape[0] < m.shape[1]:   # (r, fan) — row subspace: QR(m.T)
            return torch.linalg.qr(m.T).Q
        else:                           # (fan, r) — column subspace: QR(m)
            return torch.linalg.qr(m).Q
    Q_a = _basis(a_f)   # (large_dim, r)
    Q_b = _basis(b_f)   # (large_dim, r)
    M = Q_a.T @ Q_b     # (r, r)
    sigma = torch.linalg.svd(M, full_matrices=False).S
    return sigma[0].clamp(0.0, 1.0).item()


METRICS = {
    'cosine':        cosine_sim,
    'frobenius':     gram_frobenius,
    'principal':     max_principal_cos,
    'frobenius_ab':  gram_frobenius_delta,
}

METRIC_LABELS = {
    'cosine':        'Cosine similarity (flattened)',
    'frobenius':     'Gram Frobenius norm / r',
    'principal':     'Max principal angle cosine',
    'frobenius_ab':  'Gram Frobenius norm (B@A)',
}


# ── Matrix-level analysis ─────────────────────────────────────────────────────

def task_similarity_matrix(adapters: dict, mode: str = 'A',
                           metric: str = 'cosine') -> np.ndarray:
    """Build an (n_tasks × n_tasks) similarity matrix averaged across layers.

    Args:
        adapters: {task_id: {layer_name: {'A': Tensor, 'B': Tensor}}}
        mode:     'A'  — compare A matrices directly
                  'B'  — compare B matrices directly
                  'AB' — compare B@A low-rank update matrices
        metric:   'cosine' | 'frobenius' | 'principal'

    Returns:
        sim_matrix: np.ndarray shape (n_tasks, n_tasks)
    """
    num_tasks   = len(adapters)
    layer_names = list(adapters[0].keys())
    sim_matrix  = np.zeros((num_tasks, num_tasks))
    sim_fn      = METRICS[metric]

    for i in range(num_tasks):
        for j in range(num_tasks):
            sims = []
            for layer in layer_names:
                if layer not in adapters[i] or layer not in adapters[j]:
                    continue
                if mode == 'A':
                    m_i = adapters[i][layer]['A']
                    m_j = adapters[j][layer]['A']
                elif mode == 'B':
                    m_i = adapters[i][layer]['B']
                    m_j = adapters[j][layer]['B']
                else:
                    m_i = delta_W(adapters[i][layer])
                    m_j = delta_W(adapters[j][layer])
                sims.append(sim_fn(m_i, m_j))
            sim_matrix[i, j] = float(np.mean(sims)) if sims else 0.0

    return sim_matrix


def mean_off_diagonal(matrix: np.ndarray) -> float:
    """Average similarity between distinct task pairs (i ≠ j)."""
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(matrix[mask].mean())


def plot_similarity_heatmap(sim_matrix: np.ndarray, title: str,
                            task_names=None, ax=None):
    """Annotated heatmap of a task-pair similarity matrix."""
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

    vmin = sim_matrix.min() if sim_matrix.min() < 0 else 0

    if use_seaborn:
        import seaborn as sns
        sns.heatmap(
            sim_matrix, annot=True, fmt='.3f',
            cmap='RdYlGn_r', vmin=vmin, vmax=sim_matrix.max(),
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
    else:
        im = ax.imshow(sim_matrix, cmap='RdYlGn_r', vmin=vmin, vmax=sim_matrix.max())
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha='right')
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
