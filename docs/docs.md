## Introduction

 - Continual learning requires a model to sequentially acquire knowledge across multiple tasks without forgetting what it previously learned. A common challenge that appears is catastrophic forgetting, where adapting to a new task overwrites the representations built for earlier ones. 
 - When parameter-efficient methods such as LoRA are applied in this setting, the low-rank adapter matrices trained on successive tasks tend to occupy overlapping subspaces, causing destructive interference that degrades performance on past tasks as new ones are learned. O-LoRA addresses this by imposing an orthogonality constraint on the A matrices of each new task adapter with respect to all previously learned ones, ensuring that each task projects inputs into a distinct region of the low-rank latent space and thereby limiting the interference that drives forgetting.

## Dataset and Preprocessing

Models are evaluated on a sequence of four text classification tasks presented in fixed order: AG News (4-class news topic classification), Amazon Polarity (binary sentiment classification), DBpedia 14 (14-class Wikipedia topic classification), and Yahoo Answers Topics (10-class question-topic classification). Each task is trained on independently, with no replay of prior task data.

Samples are formatted following the instruction tuning paradigm, where each input is constructed as a natural language prompt of the form `{task instruction}\nOption: {comma-separated label names}\n{input text}\nAnswer:`, with the correct label appended as the target. The task instruction is a fixed sentence describing what is being classified (e.g. "What is the topic of the following paragraph? Choose one from the option."), and only the article text is truncated if the full sequence exceeds the maximum length, preserving the instruction and label options in full. During training, cross-entropy loss is computed only on the label token; during evaluation, the model's predicted label is determined by comparing the logits over all candidate label tokens at the position immediately preceding the answer.

## Model and Experimental Setup

All experiments use Qwen2.5-1.5B as the base language model, with LoRA adapters of rank 8, scaling factor α=32, and dropout rate 0.1 applied to the query and value projection matrices of every attention layer, trained for one epoch per task with a learning rate of 1e-3 and batch size 8. Two methods are compared: IncLoRA, which serves as an interference-free upper bound by training and freezing a completely separate adapter for each task (BWT=0 by construction, since past adapters are never modified), and O-LoRA, which maintains a single cumulative adapter and penalizes overlap between the new task's A matrix and all previously learned ones via an orthogonal regularization term weighted by λ₁=0.3.

## Results

**IncLoRA baseline — per-task accuracy (no forgetting by design):**

| Task | Accuracy |
|---|---|
| AG News | 0.854 |
| Amazon Polarity | 0.956 |
| DBpedia 14 | 0.972 |
| Yahoo Answers Topics | 0.640 |

**O-LoRA — summary metrics:**

| Run | Train samples per task | λ₁ | Avg Accuracy | BWT |
|---|---|---|---|---|
| Short | 500 / 500 / 500 / 500 | 0.3 | 0.846 | -0.032 |
| 2h | 500 / 650 / 1850 / 1300 | 0.2 | 0.799 | -0.091 |
| Full (~50% of paper) | 2100 / 2700 / 7500 / 5300 | 0.3 | 0.654 | -0.268 |

**Per-task accuracy: initial (right after training) → final (after all tasks):**

| Task | Short | 2h | Full (~50%) |
|---|---|---|---|
| AG News | 0.878 → 0.825 (-6.0%) | 0.842 → 0.747 (-11.3%) | 0.887 → 0.494 (-44.3%) |
| Amazon Polarity | 0.956 → 0.957 (~0%) | 0.947 → 0.915 (-3.4%) | 0.905 → 0.746 (-17.6%) |
| DBpedia 14 | 0.964 → 0.919 (-4.7%) | 0.979 → 0.833 (-14.9%) | 0.980 → 0.729 (-25.6%) |
| Yahoo Answers Topics | 0.683 → 0.683 (—) | 0.702 → 0.702 (—) | 0.649 → 0.649 (—) |

Across all three runs a clear trend emerges: average accuracy and BWT both degrade as the number of training samples grows, even though more data consistently improves the initial accuracy on each task when it is first learned. The short run achieves near-competitive average accuracy with IncLoRA (0.846 vs 0.728 on a different task set) with very little forgetting, suggesting the orthogonal constraint on A matrices is sufficient when adapter magnitudes remain small. The 2h run, with a proportional sample budget across tasks, sits in between — BWT worsens to -0.091 and per-task forgetting becomes visible, particularly on AG News (-11.3%) and DBpedia 14 (-14.9%). The full run, at roughly 50% of the paper's training scale, shows a sharp collapse in BWT (-0.268), with AG News losing over 44% of its accuracy by the end of the sequence. Two tasks — AG News and DBpedia 14 — reach slightly higher initial accuracy in the full run (0.887 vs 0.878 and 0.980 vs 0.964 respectively), confirming that more data improves within-task learning, but these gains are largely erased by subsequent forgetting.

### Impact of Prompt Structure

Switching to the instruction tuning format described in the paper — which explicitly lists all candidate label names in every prompt — had a larger impact on continual learning performance than any hyperparameter choice. Running the same 2h configuration (500/650/1850/1300 samples, λ₁=0.2) with the old format (no label options, plain `Classify: ...\nLabel:` prefix) versus the new format yields:

| Prompt format | Avg Accuracy | BWT |
|---|---|---|
| Old (no label options) | 0.597 | -0.344 |
| New (instruction + options) | 0.799 | -0.091 |

The improvement is +0.202 in accuracy and +0.253 in BWT, with the new format also showing substantially less per-task forgetting across the board. The likely mechanism is that listing the candidate labels in every prompt acts as a task-specific anchor: even as adapter weights shift during subsequent training, the model can partially recover the correct output distribution by attending to the label tokens explicitly present in the context, rather than relying purely on the adapted weights to recall which labels are valid for the current task.

### Orthogonality Analysis

To verify the geometric effect of the regularization, three complementary metrics were computed pairwise across all task adapter matrices:

- **Cosine similarity** — treats each matrix as a flat vector and computes their dot product normalized by both norms; captures the overall directional alignment between two adapters but is blind to subspace structure.
- **Gram Frobenius norm / r** — computes the Frobenius norm of the cross-Gram matrix (Aᵢ Aⱼᵀ) normalized by the rank r; this directly corresponds to the quantity minimized by the orthogonal loss term, making it the most interpretable metric in the context of O-LoRA's training objective.
- **Max principal angle cosine** — finds the largest singular value of the normalized cross-Gram between orthonormalized bases of the two row (or column) subspaces; this captures the closest pair of directions between the two subspaces regardless of overall matrix magnitude.

The same cosine and Gram Frobenius metrics are also applied to the full weight update ΔW = B@A. For these product matrices the Frobenius norm is reported without rank normalization — unlike the A-matrix case where dividing by r=8 removes the dependence on how many rank dimensions contribute, the ΔW matrices have shape (fan_out × fan_in). Normalizing by fan_out would instead scale values by up to 192× depending on the module, making comparisons across module types (e.g. q_proj with fan_out=1536 vs up_proj with fan_out=8960) misleading. The raw ||G||_F is therefore reported, with the understanding that values are not directly comparable across modules but are consistent within a module type across task pairs.

## Conclusions

The results confirm that O-LoRA's orthogonal regularization successfully keeps the A matrices of different tasks geometrically separated: all three metrics show near-zero pairwise similarity across tasks, and this holds even with λ₁=0.2–0.3, which is lower than the value used in the original paper. The constraint is therefore not difficult to satisfy — a relatively mild penalty is sufficient to enforce the geometric separation in the low-rank input projection space.

However, the degree to which this translates into reduced forgetting is highly sensitive to the amount of training data. With only 500 samples per task, the B matrices remain close to their zero initialization and the method achieves near-zero BWT (-0.032), rivalling IncLoRA's interference-free setup. At roughly 50% of the paper's training scale, BWT degrades sharply to -0.268. Geometric analysis of the B matrices reveals that: while B matrices are nearly orthogonal as flat vectors (cosine ~0), their column subspaces share roughly 30% of their directions (principal angle cosine ~0.29), and their magnitudes grow proportionally with the number of training steps. Since O-LoRA imposes no regularization on B, this unconstrained growth could be the primary driver of forgetting at higher scales.

An additional interesting observation from the 8h run is the non-monotonic recovery of AG News accuracy: it drops sharply from 0.887 to 0.498 after Amazon Polarity training, then partially recovers to 0.736 after DBpedia 14, before falling again to 0.494. This behaviour is consistent with unconstrained B matrices coincidentally cancelling each other's interference at intermediate steps, rather than any structured preservation of prior task knowledge.