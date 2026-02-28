# Research References & Resources

> All papers, repos, datasets, and implementation details gathered for Phase 1 & 2.

---

## Papers

### O-LoRA — Orthogonal Subspace Learning for Language Model Continual Learning
- **Venue:** EMNLP 2023 Findings
- **Authors:** Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, Xuanjing Huang
- **ArXiv:** https://arxiv.org/abs/2310.14152
- **ACL Anthology:** https://aclanthology.org/2023.findings-emnlp.715/
- **Official Repo:** https://github.com/cmnfriend/O-LoRA
- **Base Models:** T5-Large (~770M), LLaMA-2

#### How O-LoRA Works

Uses **two sets of LoRA matrices per layer**:
- `lora_A` / `lora_B` — frozen, accumulated subspaces from all **previous** tasks (rank = `r_sum`)
- `loranew_A` / `loranew_B` — trainable matrices for the **current** task (rank = `r`)

**Forward pass:**
```python
# Previous tasks (frozen)
result += lora_B(lora_A(x)) * scaling
# Current task (trainable)
result += loranew_B(loranew_A(x)) * scaling
```

**Orthogonality loss** (enforced on A matrices):
```python
L_orth = SUM over all layers of |lora_A * loranew_A^T|_1
# lora_A: [r_sum, d], loranew_A: [r, d]
# Product: [r_sum, r] — should be zero matrix if orthogonal
```

**Total loss:**
```
L_total = L_task + lambda_1 * L_orth + lambda_2 * L_l2
# Default: lambda_1 = 0.5, lambda_2 = 0
```

**After each task:** `loranew_A` is concatenated into `lora_A` (dim 0), `loranew_B` into `lora_B` (dim 1). `r_sum` grows by `r` per task.

#### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 8 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| Learning rate | 1e-3 |
| LR scheduler | constant |
| Epochs per task | 1 |
| Batch size | 8 per device |
| Max source length | 512 |
| Max target length | 50 |
| lambda_1 (orth loss) | 0.5 |
| lambda_2 (L2 reg) | 0 |
| Target modules | q, v projections |
| DeepSpeed | Stage 2 |

---

### ELLA — Efficient Lifelong Learning for Adapters in LLMs
- **Venue:** EACL 2026 (also NeurIPS 2025 Workshop on CCFM)
- **Authors:** Shristi Das Biswas, Yue Zhang, Anwesan Pal, Radhika Bhargava, Kaushik Roy
- **Affiliations:** Purdue University, AWS / Amazon
- **ArXiv:** https://arxiv.org/abs/2601.02232
- **HTML:** https://arxiv.org/html/2601.02232v2
- **OpenReview:** https://openreview.net/forum?id=A0XWBtBfFU
- **Amazon Science:** https://www.amazon.science/publications/ella-efficient-lifelong-learning-for-adapters-in-large-language-models
- **Project Website:** https://sites.google.com/view/ella-llm/home
- **Official Repo:** **No public code yet** (project site says "Coming Soon")
- **Base Models:** T5-Base (220M), T5-Large (770M), T5-XL (3B), LLaMA-3.1-8B

#### How ELLA Works

Regularization-based CL framework using **selective subspace de-correlation**:

1. **Accumulate past updates** into a single matrix `W_past` per layer (not per task)
2. **Decompose** `W_past` via SVD into high-energy (task-specific, protected) and low-energy (reusable) directions
3. **Anisotropic regularization:** penalize new updates that align with high-energy past directions:
   ```
   L_ELLA = ||DeltaW_t * W_past||_F^2
   ```
4. **Closed-form solution** acts as anisotropic shrinkage:
   ```
   (DeltaW_t*)_ij = G_ij / (1 + lambda * E_ij^2)
   ```
   High-energy past directions get strongly shrunk; low-energy directions pass through freely.

#### Key Properties
- No data replay
- No architecture expansion (single LoRA adapter, not one per task)
- No task labels at inference
- Negligible storage: **4.19 MB** vs O-LoRA's 31.46 MB (8x smaller)
- 35x smaller memory footprint than comparable methods

#### Why ELLA > strict orthogonality (O-LoRA)
Strict orthogonality exhausts available low-rank capacity fast and prevents beneficial knowledge transfer. ELLA's soft de-correlation allows partial overlap in low-energy directions, enabling forward transfer while protecting task-specific knowledge.

#### Results Highlights
| Benchmark | ELLA | Improvement |
|-----------|------|-------------|
| Standard CL (T5-Large) | 79.9% | vs 76.7% for LB-CL |
| Long Sequence | 73.6% | +4.3% over prior best |
| TRACE | 40.0% OA | +23.3 over DATA |

---

### LoRA — Low-Rank Adaptation of Large Language Models
- **Venue:** ICLR 2022
- **ArXiv:** https://arxiv.org/abs/2106.09685
- **Official Repo:** https://github.com/microsoft/LoRA
- **HuggingFace PEFT:** https://github.com/huggingface/peft

---

### IncLoRA — Incremental LoRA (Baseline)

Not a standalone paper — it's a standard baseline used across CL literature.

**Definition:** Add a new LoRA adapter per task, freeze previous adapters. No orthogonality constraint, no regularization. Subspace separation is naive.

**Implementation with HuggingFace PEFT:**
```python
from peft import LoraConfig, get_peft_model

for task_id, task_data in enumerate(tasks):
    if task_id == 0:
        model = get_peft_model(base_model, lora_config, adapter_name=f"task_{task_id}")
    else:
        model.add_adapter(f"task_{task_id}", lora_config)

    model.set_adapter(f"task_{task_id}")
    train(model, task_data)
    model.save_pretrained(f"./adapters/task_{task_id}")
```

**Alternative (merge-and-continue):**
```python
model = PeftModel.from_pretrained(base_model, "./adapters/task_0")
model = model.merge_and_unload()  # merge LoRA into base weights
model = get_peft_model(model, lora_config)  # fresh LoRA for next task
```

**Relationship to other baselines:**
| Baseline | Definition |
|----------|-----------|
| SeqFT | Fine-tune ALL parameters sequentially. Severe forgetting. |
| SeqLoRA | Fine-tune a single fixed LoRA across tasks. Also forgets. |
| IncLoRA | New LoRA per task, freeze old ones. Naive subspace separation. |
| O-LoRA | IncLoRA + orthogonality constraints. |
| ELLA | Single LoRA + anisotropic regularization on past directions. |

---

## Datasets — Standard CL Benchmark

Used by both O-LoRA and ELLA as primary evaluation.

| Dataset | HuggingFace ID | Task Type | Classes | Train Size |
|---------|---------------|-----------|---------|------------|
| AG News | `ag_news` | News topic classification | 4 | 120,000 |
| Yelp | `yelp_review_full` | Sentiment classification | 5 | 650,000 |
| Amazon | `amazon_polarity` | Sentiment classification | 2 | 3,600,000 |
| DBPedia | `dbpedia_14` | Entity classification | 14 | 560,000 |
| Yahoo Answers | `yahoo_answers_topics` | Topic classification | 10 | 1,400,000 |

**Task orderings (from papers):**
- Order 1: DBPedia → Amazon → Yahoo → AG News
- Order 2: DBPedia → Amazon → AG News → Yahoo
- Order 3: Yahoo → Amazon → AG News → DBPedia

Papers evaluate across **3 orderings** to reduce ordering bias.

### Other Benchmarks

**Long Sequence Benchmark (15 tasks):**
Extends the standard 5 with GLUE/SuperGLUE tasks (SST-2, MRPC, QQP, MNLI, QNLI, RTE, WNLI, CoLA) + IMDB.

**TRACE Benchmark (8 tasks):**
C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten.
- Repo: https://github.com/BeyonderXX/TRACE

---

## Metrics — How to Evaluate

### The Accuracy Matrix

Build a **T x T matrix** `R[i][j]` = accuracy on task j after training through task i.

```
         Task 1   Task 2   Task 3   Task 4   Task 5
After T1:  R[1,1]   -        -        -        -
After T2:  R[2,1]   R[2,2]   -        -        -
After T3:  R[3,1]   R[3,2]   R[3,3]   -        -
After T4:  R[4,1]   R[4,2]   R[4,3]   R[4,4]   -
After T5:  R[5,1]   R[5,2]   R[5,3]   R[5,4]   R[5,5]
```

### Formulas

**Average Accuracy (ACC):**
```
ACC = (1/T) * SUM_{j=1}^{T} R[T, j]
```

**Backward Transfer (BWT):**
```
BWT = (1/(T-1)) * SUM_{j=1}^{T-1} (R[T, j] - R[j, j])
```
- BWT < 0 → forgetting
- BWT = 0 → no forgetting
- BWT > 0 → positive backward transfer (rare)

**Forward Transfer (FWT):**
```
FWT = (1/(T-1)) * SUM_{j=2}^{T} (R[j, j] - b_j)
```
Where b_j is zero-shot baseline on task j.

### Implementation
```python
import numpy as np

accuracy_matrix = np.zeros((num_tasks, num_tasks))

for train_task in range(num_tasks):
    # train on train_task ...
    for eval_task in range(train_task + 1):
        accuracy_matrix[train_task][eval_task] = evaluate(model, datasets[eval_task])

T = num_tasks
ACC = np.mean(accuracy_matrix[T-1, :])
BWT = np.mean([accuracy_matrix[T-1, j] - accuracy_matrix[j, j] for j in range(T-1)])
```

---

## Key GitHub Repositories

### Primary References (use these)
| Repo | What | URL |
|------|------|-----|
| O-LoRA | Full CL pipeline, our main codebase reference | https://github.com/cmnfriend/O-LoRA |
| HuggingFace PEFT | LoRA library, multi-adapter support | https://github.com/huggingface/peft |
| TRACE | 8-task benchmark with training scripts | https://github.com/BeyonderXX/TRACE |

### Related CL + LoRA Methods
| Repo | Method | Venue |
|------|--------|-------|
| [InfLoRA](https://github.com/liangyanshuo/InfLoRA) | Interference-free LoRA | CVPR 2024 |
| [SD-LoRA](https://github.com/WuYichen-97/SD-Lora-CL) | Scalable decoupled LoRA | ICLR 2025 Oral |
| [TreeLoRA](https://github.com/ZinYY/TreeLoRA) | Layer-wise LoRA with gradient tree | ICML 2025 |
| [KeepLoRA](https://github.com/MaolinLuo/KeepLoRA) | Residual gradient adaptation | 2025 |
| [Online-LoRA](https://github.com/Christina200/Online-LoRA-official) | Task-free online CL | WACV 2025 |
| [continual-lora](https://github.com/luk-st/continual-lora) | LoRA weight init & merging under CL | — |
| [FastKGE](https://github.com/seukgcode/FastKGE) | IncLoRA for knowledge graphs | IJCAI 2024 |

### Surveys & Awesome Lists
| Repo | Description |
|------|-------------|
| [llm-continual-learning-survey](https://github.com/Wang-ML-Lab/llm-continual-learning-survey) | ACM Computing Surveys 2025, categorized methods |
| [awesome-lifelong-learning-methods-for-llm](https://github.com/zzz47zzz/awesome-lifelong-learning-methods-for-llm) | 12 CL scenarios, regularly updated |
| [Awesome-LoRAs](https://github.com/ZJU-LLMs/Awesome-LoRAs) | Curated list of LoRA variants |
| [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning) | Broader incremental/continual learning |

### Full Pipeline Frameworks
| Repo | Description |
|------|-------------|
| [ContinualLM](https://github.com/UIC-Liu-Lab/ContinualLM) | 6-domain framework, 10+ methods |
| [LlamaFactory](https://github.com/hiyouga/LlamaFactory) | Unified fine-tuning for 100+ LLMs |
| [GMvandeVen/continual-learning](https://github.com/GMvandeVen/continual-learning) | 12 CL methods implemented |

---

## HuggingFace PEFT Documentation

- [LoRA Developer Guide](https://huggingface.co/docs/peft/main/en/developer_guides/lora) — `add_adapter`, `set_adapter`, `disable_adapter`
- [Model Merging Guide](https://huggingface.co/docs/peft/en/developer_guides/model_merging) — TIES, DARE merging
- [Mixed Adapter Types](https://huggingface.co/docs/peft/en/developer_guides/mixed_models) — multiple adapter types
- [PEFT + Transformers](https://huggingface.co/docs/transformers/en/peft) — `load_adapter()`, hotswap
- [PEFT Course](https://huggingface.co/learn/smol-course/unit1/3a) — HuggingFace smol course
- [PEFT CL Discussion](https://github.com/huggingface/peft/issues/184) — continual training patterns

---

## Important Warnings

1. **HuggingFace PEFT has a different "OLoRA"** — `init_lora_weights="olora"` is QR-decomposition init by Kerim Buyukakyuz ([arxiv.org/abs/2406.01775](https://arxiv.org/abs/2406.01775)). This is **NOT** the continual learning O-LoRA paper. Don't confuse them.

2. **ELLA has no public code** — we'll need to implement it from the paper description. The core mechanism (anisotropic regularization on accumulated `W_past`) is well-described.

3. **Old ELLA repos are unrelated** — [github.com/Lifelong-ML/ELLA](https://github.com/Lifelong-ML/ELLA) is a 2013 multi-task learning algorithm. [github.com/TencentQQGYLab/ELLA](https://github.com/TencentQQGYLab/ELLA) is a diffusion model paper. Neither is related to the 2026 LLM adapter paper.

---

## Practical Strategy

1. **Start from the O-LoRA repo** — it has the full CL pipeline, datasets, training scripts for T5-Large and LLaMA-2
2. **IncLoRA = O-LoRA with `lambda_1=0`** — just disable the orthogonality loss
3. **Implement ELLA from scratch** — add the anisotropic regularization term into the same training loop
4. **Use the Standard CL Benchmark** (5 datasets, 3 orderings) — matches both papers
