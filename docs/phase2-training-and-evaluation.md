# Phase 2: Sequential Training & Performance Evaluation

> This phase achieves the **80% grade** requirement.

## Objective

Train Ella and the IncLoRA baseline on a standard Continual Learning setup based on 5 NLP tasks. Evaluate the performance of Ella vs IncLoRA by measuring overall accuracy and backward transfer.

---

## Step 1: Select 5 NLP Tasks

Choose 5 diverse NLP tasks for the continual learning benchmark. The tasks should be sufficiently different to test whether the methods can learn sequentially without catastrophic forgetting.

- [ ] Research standard CL benchmarks used in the Ella and OLoRA papers
- [ ] Select 5 tasks (e.g., sentiment analysis, NLI, summarization, QA, translation — finalize based on what the papers use)
- [ ] Prepare datasets using HuggingFace `datasets` library
- [ ] Implement a unified data loading pipeline that works for all 5 tasks
- [ ] Define evaluation metrics per task (accuracy, F1, ROUGE, BLEU — depends on task type)

**Key files to create:**
- `src/data/task_loader.py` — unified data loading for all 5 tasks
- `configs/tasks.yaml` — task definitions, dataset names, splits, metrics

## Step 2: Sequential Training Loop

Train each method sequentially: Task 1 → Task 2 → Task 3 → Task 4 → Task 5, saving adapter weights after each task.

- [ ] Implement the continual learning training loop:
  ```
  for task in [task1, task2, task3, task4, task5]:
      train(model, adapter, task)
      save_adapter_weights(adapter, task_id)
      evaluate_all_tasks(model, adapter)  # measure backward transfer
  ```
- [ ] Run the full loop for **IncLoRA**
- [ ] Run the full loop for **Ella**
- [ ] Run the full loop for **OLoRA**
- [ ] Save all adapter checkpoints (we need these for Phase 3)

**Key files to create:**
- `src/training/continual_trainer.py` — the sequential training loop
- `scripts/run_training.py` — entry point to launch training for any method

## Step 3: Performance Evaluation

- [ ] After training, evaluate each method on all 5 tasks
- [ ] Compute **overall accuracy** per task per method
- [ ] Compute **backward transfer**: how much does Task N performance degrade after learning subsequent tasks?
  ```
  BWT = (1 / (T-1)) * sum(R_{T,i} - R_{i,i}) for i = 1..T-1
  ```
  where R_{T,i} = accuracy on task i after training on all T tasks, R_{i,i} = accuracy on task i right after training on it.
- [ ] Compare Ella vs IncLoRA (and OLoRA) in a results table

**Key files to create:**
- `src/evaluation/metrics.py` — accuracy, backward transfer computation
- `src/evaluation/evaluate.py` — run evaluation across all tasks and methods

---

## Expected Output

A results table like:

| Method  | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Acc | BWT   |
|---------|--------|--------|--------|--------|--------|---------|-------|
| IncLoRA | ...    | ...    | ...    | ...    | ...    | ...     | ...   |
| OLoRA   | ...    | ...    | ...    | ...    | ...    | ...     | ...   |
| Ella    | ...    | ...    | ...    | ...    | ...    | ...     | ...   |

---

## Definition of Done

- All three methods have been trained sequentially on all 5 tasks.
- Accuracy and backward transfer are computed and recorded.
- Ella vs IncLoRA comparison is clear and documented.
- All adapter checkpoints are saved for Phase 3 analysis.
