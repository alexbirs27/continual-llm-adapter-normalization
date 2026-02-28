# Roadmap: Continual Learning for LLMs using Adapter-Based Methods (3)

> **Research Question:** Does OLoRA truly keep new task updates orthogonal to old ones?
> Explore the geometry of the OLoRA Continual Learning method.

---

## Scoring Criteria

| Grade | Requirements |
|-------|-------------|
| **80%** | Implement IncLoRA and OLoRA baselines and measure overall accuracy and backward transfer |
| **100%** | Test orthogonality of A_t vs A_i and A_t·B_t vs A_i·B_i; analyse orthogonality per layer and per module |

---

## Resources

- **OLoRA** — Orthogonal Subspace Learning for Language Model Continual Learning
- **LoRA** — Low-Rank Adaptation of Large Language Models
- **Ella** — Efficient Lifelong Learning for Adapters in LLMs

---

## Phase 1: Environment Setup & Baseline Implementations

> **Goal:** Set up the project environment and implement all three adapter methods.

- Set up PEFT environment and select base LLM
- Implement **IncLoRA** (Incremental LoRA — control baseline)
- Implement **OLoRA** (Orthogonal LoRA)
- Implement **Ella** (Efficient Lifelong Learning for Adapters)

See: [Phase 1 Details](./phase1-setup-and-baselines.md)

---

## Phase 2: Sequential Training & Performance Evaluation (80% grade)

> **Goal:** Train Ella and IncLoRA on a standard CL setup with 5 NLP tasks.
> Evaluate performance of Ella vs IncLoRA. Measure accuracy and backward transfer.

- Select 5 NLP tasks for the continual learning setup
- Train Ella and the IncLoRA baseline sequentially on all 5 tasks
- Evaluate performance: overall accuracy per task, backward transfer
- Compare Ella vs IncLoRA results

See: [Phase 2 Details](./phase2-training-and-evaluation.md)

---

## Phase 3: Geometric Audit — Orthogonality Analysis (100% grade)

> **Goal:** Evaluate the geometry of the OLoRA updates with respect to orthogonality.

- Extract adapter weights (A and B matrices) after each task
- Test orthogonality of **A_t vs A_i** (cosine similarity of flattened weight vectors)
- Test orthogonality of **A_t·B_t vs A_i·B_i** (product-level orthogonality)
- Analyse orthogonality **per layer** — inspect if orthogonality holds for some layers more than others
- Analyse orthogonality **per module** — separately for each module (attention layer 1, mlp layer 1, att layer 2, mlp layer 2, etc.)

See: [Phase 3 Details](./phase3-geometric-audit.md)

---

## Phase 4: Analysis & Final Deliverables

> **Goal:** Compile results, generate visualizations, and write up findings.

- Benchmark table: IncLoRA vs OLoRA vs Ella on all 5 tasks
- Geometry heatmaps: orthogonality scores across all layers and modules
- Correlate orthogonality decay with accuracy loss
- Full experimental setup documentation

See: [Phase 4 Details](./phase4-analysis-and-deliverables.md)
