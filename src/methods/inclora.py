"""IncLoRA: Incremental LoRA — one adapter per task, freeze previous ones."""

import torch
from peft import get_peft_model, LoraConfig, TaskType


class IncLoRA:
    """Multi-adapter baseline using HuggingFace PEFT.

    For each new task, a fresh LoRA adapter is added and set as active.
    Previous adapters are frozen (PEFT handles this via set_adapter).
    At evaluation time, the adapter for the specific task is activated.
    """

    def __init__(self, base_model, lora_config):
        self.model = base_model
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=list(lora_config.target_modules),
        )
        self.adapters = []

    def prepare_task(self, task_id):
        """Add a new LoRA adapter for this task."""
        name = f"task_{task_id}"
        if task_id == 0:
            self.model = get_peft_model(self.model, self.peft_config, adapter_name=name)
        else:
            self.model.add_adapter(name, self.peft_config)
            self.model.set_adapter(name)
        self.adapters.append(name)

    def get_trainable_params(self):
        """Return parameters that require grad (current adapter only)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_loss(self, input_ids, attention_mask, labels):
        """Standard causal LM loss."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def after_task(self, task_id):
        """Freeze current adapter by disabling its gradients."""
        for name, param in self.model.named_parameters():
            if f"task_{task_id}" in name:
                param.requires_grad = False

    def set_eval_adapter(self, task_id):
        """Activate the adapter for a specific task (for evaluation)."""
        self.model.set_adapter(f"task_{task_id}")

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
