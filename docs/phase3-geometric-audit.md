# Phase 3: Geometric Audit — Orthogonality Analysis

> This phase achieves the **100% grade** requirement.

## Objective

Evaluate the geometry of the OLoRA updates with respect to orthogonality. Extract adapter weights after each task and measure how orthogonal the updates truly are — at the weight level, at the product level, per layer, and per module.

---

## Step 1: Extract Adapter Weights

After the sequential training in Phase 2, we have saved adapter checkpoints after each task. Now extract the raw A and B matrices.

- [ ] Load saved adapter checkpoints for all tasks (Task 1 through Task 5)
- [ ] For each task t and each layer, extract:
  - **A_t** — the A matrix of the LoRA adapter
  - **B_t** — the B matrix of the LoRA adapter
- [ ] Store extracted weights in a structured format for analysis
- [ ] Do this for all three methods (IncLoRA, OLoRA, Ella)

**Key files to create:**
- `src/analysis/weight_extraction.py` — extract A and B matrices from checkpoints

## Step 2: Test Orthogonality of A_t vs A_i

Measure how orthogonal the A matrices are between different tasks using cosine similarity of the flattened weight vectors.

- [ ] For each pair of tasks (t, i) where t != i:
  - Flatten A_t and A_i into vectors
  - Compute cosine similarity: `cos_sim = (A_t · A_i) / (||A_t|| * ||A_i||)`
- [ ] A score of **0** means perfectly orthogonal (success)
- [ ] A score of **1** means fully overlapping (failure / interference)
- [ ] Generate a **task-pair similarity matrix** (5x5 heatmap) for each method
- [ ] Compare: does OLoRA achieve lower cosine similarity than IncLoRA and Ella?

**Key files to create:**
- `src/analysis/orthogonality.py` — cosine similarity computation between adapter matrices

## Step 3: Test Orthogonality of A_t·B_t vs A_i·B_i

The product A·B represents the actual low-rank update applied to the model's hidden states. This is a more meaningful measure than looking at A or B alone.

- [ ] For each pair of tasks (t, i):
  - Compute the products: `delta_t = A_t @ B_t` and `delta_i = A_i @ B_i`
  - Flatten and compute cosine similarity
- [ ] Generate a product-level task-pair similarity matrix
- [ ] Compare across methods: is product-level orthogonality better or worse than A-level orthogonality?

## Step 4: Per-Layer Orthogonality Analysis

Don't just average across the whole model. Inspect whether orthogonality holds more in some layers than others.

- [ ] For each layer l in the model (e.g., layers 0–31 for a 32-layer model):
  - Compute the A_t vs A_i cosine similarity for that specific layer
  - Compute the A_t·B_t vs A_i·B_i cosine similarity for that specific layer
- [ ] Generate a **layer-wise orthogonality plot**: x-axis = layer index, y-axis = average cosine similarity across task pairs
- [ ] Identify patterns:
  - Do early layers maintain orthogonality better than later layers?
  - Are there specific "bottleneck" layers where orthogonality collapses?
  - Does this pattern differ between OLoRA and IncLoRA?

## Step 5: Per-Module Orthogonality Analysis

Analyse orthogonality separately for each module type within each layer.

- [ ] For each layer, separate the adapter weights by module:
  - **Attention modules**: Q, K, V, and O projections
  - **MLP modules**: up projection, down projection, gate projection
- [ ] Compute orthogonality metrics separately for:
  - Attention layer 1, MLP layer 1
  - Attention layer 2, MLP layer 2
  - ... and so on for all layers
- [ ] Generate per-module orthogonality heatmaps
- [ ] Answer: do attention modules preserve orthogonality differently from MLP modules?

**Key files to create:**
- `src/analysis/per_layer_analysis.py` — layer-wise and module-wise orthogonality computation
- `scripts/run_analysis.py` — entry point to run the full geometric audit

---

## Expected Outputs

1. **Task-pair similarity matrices** (5x5 heatmaps) for A_t vs A_i — one per method
2. **Task-pair similarity matrices** for A_t·B_t vs A_i·B_i — one per method
3. **Layer-wise orthogonality plot** — showing how orthogonality varies across layers
4. **Module-wise orthogonality heatmap** — showing attention vs MLP per layer
5. Summary statistics: mean orthogonality score per method, per layer range, per module type

---

## Definition of Done

- Orthogonality of A_t vs A_i is computed and visualized for all methods.
- Orthogonality of A_t·B_t vs A_i·B_i is computed and visualized for all methods.
- Per-layer analysis shows which layers maintain or lose orthogonality.
- Per-module analysis shows whether attention and MLP modules behave differently.
- Results clearly demonstrate whether OLoRA's orthogonality claims hold in practice.
