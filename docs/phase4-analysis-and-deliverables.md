# Phase 4: Analysis & Final Deliverables

## Objective

Compile all results from Phase 2 (performance) and Phase 3 (geometry) into a coherent analysis. Generate final visualizations and write up findings for the project deliverable.

---

## Step 1: Benchmark Table

- [ ] Compile the final performance comparison table:

| Method  | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Acc | BWT   |
|---------|--------|--------|--------|--------|--------|---------|-------|
| IncLoRA | ...    | ...    | ...    | ...    | ...    | ...     | ...   |
| OLoRA   | ...    | ...    | ...    | ...    | ...    | ...     | ...   |
| Ella    | ...    | ...    | ...    | ...    | ...    | ...     | ...   |

- [ ] Include per-task accuracy after full training and backward transfer scores

## Step 2: Geometry Heatmaps

- [ ] Generate final publication-quality heatmaps:
  - A_t vs A_i similarity matrices (one per method, side by side)
  - A_t·B_t vs A_i·B_i similarity matrices (one per method, side by side)
  - Layer-wise orthogonality comparison (all methods on one plot)
  - Module-wise breakdown (attention vs MLP, per layer)

**Key files to create:**
- `src/visualization/heatmaps.py` — generate all heatmap figures
- `src/visualization/plots.py` — line plots for layer-wise analysis

## Step 3: Correlate Orthogonality with Performance

- [ ] Plot orthogonality decay vs accuracy loss:
  - x-axis: average cosine similarity (orthogonality loss) at a given training step
  - y-axis: backward transfer (accuracy degradation) at the same step
- [ ] Determine if there is a clear correlation: when orthogonality degrades, does forgetting increase?
- [ ] Compare this correlation across methods

## Step 4: Full Experimental Setup Documentation

- [ ] Document the complete experimental setup:
  - Base model used and why
  - Hyperparameters for each method (rank, learning rate, alpha, etc.)
  - Task order and dataset details
  - Hardware and training time
  - Random seeds and reproducibility notes

## Step 5: Write-Up

- [ ] Summarize key findings:
  - Does OLoRA maintain orthogonality better than IncLoRA?
  - Does Ella outperform IncLoRA on backward transfer?
  - Which layers/modules are most sensitive to orthogonality collapse?
  - Is there a measurable link between orthogonality and forgetting?
- [ ] Answer the research question: **Does OLoRA truly keep new task updates orthogonal to old ones?**

---

## Final Deliverables Checklist

- [ ] **Benchmark Table** — IncLoRA vs OLoRA vs Ella on 5 tasks (accuracy + BWT)
- [ ] **Geometry Heatmaps** — orthogonality across all layers and modules
- [ ] **Correlation Analysis** — orthogonality decay vs accuracy loss
- [ ] **Experimental Setup** — full reproducibility documentation
- [ ] **Research Write-Up** — answering the research question with evidence

---

## Definition of Done

- All visualizations are generated and saved to `results/figures/`.
- The benchmark table is complete with final numbers.
- The analysis clearly answers whether OLoRA's orthogonality holds in practice.
- The experimental setup is fully documented for reproducibility.
