# Phase 1: Environment Setup & Baseline Implementations

## Objective

Set up the development environment and implement the three adapter-based continual learning methods that we will compare throughout the project.

---

## Step 1: Environment Setup

- [ ] Install dependencies: `transformers`, `peft`, `datasets`, `torch`, `accelerate`
- [ ] Select a base LLM (candidates: Llama-3-8B, Mistral-7B — pick based on available compute)
- [ ] Verify the base model loads and runs inference correctly
- [ ] Set up project structure: `src/`, `configs/`, `scripts/`, `results/`
- [ ] Set up experiment tracking (Weights & Biases or simple CSV logging)

## Step 2: Implement IncLoRA (Incremental LoRA)

**What it is:** Standard LoRA applied incrementally across tasks. No orthogonality constraint — each new task simply adds low-rank weight updates on top of previous ones.

- [ ] Implement LoRA adapter injection using PEFT
- [ ] Implement the incremental training loop: for each new task, continue fine-tuning the same LoRA adapters
- [ ] Implement weight saving/loading between tasks
- [ ] Verify: train on a single task and confirm the adapter produces reasonable outputs

**Key files to create:**
- `src/methods/inclora.py` — IncLoRA adapter logic
- `configs/inclora.yaml` — hyperparameters (rank, alpha, target modules)

## Step 3: Implement OLoRA (Orthogonal LoRA)

**What it is:** LoRA with orthogonality constraints. The A and B matrices are initialized or regularized so that each new task's adapter updates are mathematically independent (orthogonal) to previous tasks.

- [ ] Read the OLoRA paper: "Orthogonal Subspace Learning for Language Model Continual Learning"
- [ ] Implement orthogonal initialization of A and B matrices
- [ ] Implement the orthogonality regularization loss (or projection step)
- [ ] Integrate into the same training loop structure as IncLoRA
- [ ] Verify: train on two tasks and confirm the orthogonal constraint is active (log the constraint loss or projection magnitude)

**Key files to create:**
- `src/methods/olora.py` — OLoRA adapter logic with orthogonality constraints
- `configs/olora.yaml` — hyperparameters

## Step 4: Implement Ella (Efficient Lifelong Learning for Adapters)

**What it is:** A continual learning method designed specifically for adapters in LLMs. Uses efficient strategies to prevent catastrophic forgetting while keeping memory overhead low.

- [ ] Read the Ella paper: "Efficient Lifelong Learning for Adapters in LLMs"
- [ ] Implement the Ella adapter method
- [ ] Integrate into the same training loop structure
- [ ] Verify: train on a single task and confirm correct behavior

**Key files to create:**
- `src/methods/ella.py` — Ella adapter logic
- `configs/ella.yaml` — hyperparameters

---

## Definition of Done

- All three methods (IncLoRA, OLoRA, Ella) can be trained on a single NLP task and produce valid outputs.
- Weight saving/loading works correctly between tasks.
- Shared training infrastructure is in place so all methods use the same data pipeline and evaluation code.
