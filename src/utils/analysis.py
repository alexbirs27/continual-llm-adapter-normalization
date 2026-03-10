"""Post-hoc orthogonality analysis between per-task adapter updates."""

import torch


def compute_pairwise_orthogonality(task_deltas, task_names):
    """Compute pairwise cosine similarity between flattened per-task ΔW vectors.

    Args:
        task_deltas: list of lists — task_deltas[t] is a list of ΔW tensors (one per layer)
        task_names: list of task name strings

    Returns:
        dict with 'cosine_matrix' (n x n) and 'mean_off_diagonal' scalar
    """
    n = len(task_deltas)
    if n < 2:
        return None

    # Flatten each task's deltas into a single vector
    flat = []
    for deltas in task_deltas:
        flat.append(torch.cat([d.flatten().float() for d in deltas]))

    cosine_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            cosine_matrix[i, j] = torch.nn.functional.cosine_similarity(
                flat[i].unsqueeze(0), flat[j].unsqueeze(0)
            ).item()

    # Mean absolute off-diagonal cosine (lower = more orthogonal)
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_off_diag = cosine_matrix[mask].abs().mean().item()

    return {
        "cosine_matrix": cosine_matrix,
        "mean_off_diagonal": mean_off_diag,
    }


def print_orthogonality_report(result, task_names):
    """Print the pairwise cosine similarity matrix."""
    if result is None:
        return

    n = len(task_names)
    mat = result["cosine_matrix"]

    print(f"\n{'='*60}")
    print("ADAPTER ORTHOGONALITY (pairwise cosine similarity)")
    print(f"{'='*60}")

    header = f"{'':>15}" + "".join(f"{name[:10]:>12}" for name in task_names)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{task_names[i][:15]:>15}"
        row += "".join(f"{mat[i, j].item():>12.4f}" for j in range(n))
        print(row)

    print(f"\nMean |cos| (off-diag): {result['mean_off_diagonal']:.4f}")
    print("(lower = more orthogonal)")
