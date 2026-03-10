"""Shared sequential training loop for continual learning methods."""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.analysis import compute_pairwise_orthogonality, print_orthogonality_report


class ContinualTrainer:
    """Trains a continual learning method on a sequence of tasks.

    Produces an R[i][j] accuracy matrix where R[i][j] is the accuracy
    on task j after training on task i.
    """

    def __init__(self, method, datasets, training_config, tokenizer):
        self.method = method
        self.datasets = datasets
        self.config = training_config
        self.tokenizer = tokenizer
        self.results_matrix = []
        self.device = next(method.model.parameters()).device

    def train_all_tasks(self):
        """Train on all tasks sequentially, evaluating after each."""
        for task_id, task_name in enumerate(self.config.task_order):
            print(f"\n{'='*60}")
            print(f"Task {task_id}: {task_name}")
            print(f"{'='*60}")

            self.method.prepare_task(task_id)
            self._train_single_task(task_id, task_name)
            self.method.after_task(task_id)

            # Evaluate on all tasks seen so far
            row = self._evaluate_all_tasks(task_id)
            self.results_matrix.append(row)

            self._print_results_row(task_id, row)

        metrics = self.compute_metrics()
        self._print_final_metrics(metrics)

        if hasattr(self.method, "get_task_deltas"):
            task_names = self.config.task_order[:len(self.results_matrix)]
            orth = compute_pairwise_orthogonality(self.method.get_task_deltas(), task_names)
            print_orthogonality_report(orth, task_names)
            if orth:
                metrics["orthogonality"] = orth["mean_off_diagonal"]

        return self.results_matrix, metrics

    def _train_single_task(self, task_id, task_name):
        """Train on a single task."""
        train_ds, _ = self.datasets[task_name]
        dataloader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        trainable_params = self.method.get_trainable_params()
        optimizer = AdamW(trainable_params, lr=self.config.lr)
        accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)

        self.method.train_mode()

        for epoch in range(self.config.epochs_per_task):
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs_per_task}")
            for step, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                loss = self.method.get_loss(input_ids, attention_mask, labels)
                loss = loss / accum_steps
                loss.backward()

                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accum_steps
                num_batches += 1
                pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

            # Handle remaining gradients
            if (step + 1) % accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    def _evaluate_all_tasks(self, current_task_id):
        """Evaluate on all tasks seen so far, return accuracy list."""
        self.method.eval_mode()
        row = []

        for eval_task_id in range(current_task_id + 1):
            task_name = self.config.task_order[eval_task_id]
            self.method.set_eval_adapter(eval_task_id)
            acc = self._evaluate_task(task_name)
            row.append(acc)

        self.method.train_mode()
        return row

    @torch.no_grad()
    def _evaluate_task(self, task_name):
        """Evaluate accuracy on a single task.

        For causal LM classification, we check if the model's next-token
        prediction at the 'Label:' position matches the correct label token.
        """
        _, eval_ds = self.datasets[task_name]
        from src.data.datasets import DATASET_REGISTRY

        cfg = DATASET_REGISTRY[task_name]
        label_names = cfg["label_names"]

        # Get first token ID for each label name
        label_token_ids = []
        for name in label_names:
            tokens = self.tokenizer.encode(f" {name}", add_special_tokens=False)
            label_token_ids.append(tokens[0])

        dataloader = DataLoader(eval_ds, batch_size=self.config.batch_size, shuffle=False)

        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            true_labels = batch["label_idx"].to(self.device)

            # Find the position right after "Label:" — this is where labels start
            # Labels in batch["labels"] have -100 for prompt, then actual tokens
            batch_labels = batch["labels"].to(self.device)

            outputs = self.method.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [batch, seq_len, vocab]

            for i in range(input_ids.shape[0]):
                # Find the first non -100 position in labels (start of target)
                label_row = batch_labels[i]
                target_positions = (label_row != -100).nonzero(as_tuple=True)[0]
                if len(target_positions) == 0:
                    continue

                # The prediction position is one before the first target token
                pred_pos = target_positions[0].item() - 1
                if pred_pos < 0:
                    continue

                pred_logits = logits[i, pred_pos]
                # Restrict to label tokens only
                label_logits = pred_logits[label_token_ids]
                predicted_label = label_logits.argmax().item()

                if predicted_label == true_labels[i].item():
                    correct += 1
                total += 1

        accuracy = correct / max(total, 1)
        return accuracy

    def compute_metrics(self):
        """Compute ACC (average final accuracy) and BWT (backward transfer)."""
        if not self.results_matrix:
            return {"acc": 0.0, "bwt": 0.0}

        n = len(self.results_matrix)
        last_row = self.results_matrix[-1]

        # ACC: average accuracy on all tasks after training on the last task
        acc = sum(last_row) / len(last_row)

        # BWT: backward transfer
        # BWT = (1 / (n-1)) * sum_{j=0}^{n-2} (R[n-1][j] - R[j][j])
        if n <= 1:
            bwt = 0.0
        else:
            bwt_sum = 0.0
            for j in range(n - 1):
                bwt_sum += last_row[j] - self.results_matrix[j][j]
            bwt = bwt_sum / (n - 1)

        return {"acc": acc, "bwt": bwt}

    def _print_results_row(self, task_id, row):
        """Print evaluation results for the current task."""
        task_names = self.config.task_order[:task_id + 1]
        print(f"\n  Results after task {task_id}:")
        for j, (name, acc) in enumerate(zip(task_names, row)):
            print(f"    {name}: {acc:.4f}")

    def _print_final_metrics(self, metrics):
        """Print final metrics and the full results matrix."""
        n = len(self.results_matrix)
        task_names = self.config.task_order[:n]

        print(f"\n{'='*60}")
        print("RESULTS MATRIX R[i][j]")
        print(f"{'='*60}")

        header = f"{'Trained on':<20}" + "".join(f"{name[:10]:>12}" for name in task_names)
        print(header)
        print("-" * len(header))

        for i, row in enumerate(self.results_matrix):
            row_str = f"{task_names[i]:<20}"
            row_str += "".join(f"{acc:>12.4f}" for acc in row)
            # Pad with empty cells for tasks not yet seen
            row_str += "".join(f"{'':>12}" for _ in range(n - len(row)))
            print(row_str)

        print(f"\nACC: {metrics['acc']:.4f}")
        print(f"BWT: {metrics['bwt']:.4f}")
