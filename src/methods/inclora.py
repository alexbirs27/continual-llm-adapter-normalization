"""IncLoRA: Incremental LoRA — one adapter per task, freeze previous ones.

Uses minLoRA's parametrization approach (copied into src/minlora).
Each task gets a fresh LoRA, which is saved after training.
At eval time, the correct task's LoRA weights are loaded.
"""

from functools import partial

import torch
from torch import nn

from src.minlora.model import (
    LoRAParametrization,
    add_lora_by_name,
    remove_lora,
)
from src.minlora.utils import (
    get_lora_params,
    get_lora_state_dict,
    load_multiple_lora,
    select_lora,
)


class IncLoRA:
    """Multi-adapter baseline using minLoRA parametrization."""

    def __init__(self, base_model, lora_config):
        self.model = base_model
        self.r = lora_config.r
        self.alpha = lora_config.alpha
        self.dropout = lora_config.dropout
        self.target_modules = list(lora_config.target_modules)
        self.saved_lora_states = []
        self.task_deltas = []

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

    def _make_lora_config(self):
        return {
            nn.Linear: {
                "weight": partial(
                    LoRAParametrization.from_linear,
                    rank=self.r,
                    lora_dropout_p=self.dropout,
                    lora_alpha=self.alpha,
                ),
            },
            nn.Embedding: {
                "weight": partial(
                    LoRAParametrization.from_embedding,
                    rank=self.r,
                    lora_dropout_p=self.dropout,
                    lora_alpha=self.alpha,
                ),
            },
        }

    def prepare_task(self, task_id):
        """Add fresh LoRA parametrization for this task."""
        try:
            remove_lora(self.model)
        except Exception:
            pass
        add_lora_by_name(self.model, self.target_modules, lora_config=self._make_lora_config())

    def get_trainable_params(self):
        return list(get_lora_params(self.model))

    def get_loss(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def after_task(self, task_id):
        """Save current LoRA state dict, then remove LoRA from model."""
        state = get_lora_state_dict(self.model)
        self.saved_lora_states.append(state)
        self._save_task_deltas(state)
        remove_lora(self.model)

    def _save_task_deltas(self, state):
        As = sorted([k for k in state if k.endswith("lora_A")])
        Bs = sorted([k for k in state if k.endswith("lora_B")])
        scaling = self.alpha / self.r
        deltas = []
        for a_key, b_key in zip(As, Bs):
            dw = (state[b_key] @ state[a_key]).cpu() * scaling
            deltas.append(dw)
        self.task_deltas.append(deltas)

    def get_task_deltas(self):
        return self.task_deltas

    def set_eval_adapter(self, task_id):
        """Load all saved LoRAs and select the one for task_id."""
        try:
            remove_lora(self.model)
        except Exception:
            pass
        add_lora_by_name(self.model, self.target_modules, lora_config=self._make_lora_config())
        load_multiple_lora(self.model, self.saved_lora_states[:task_id + 1])
        select_lora(self.model, task_id)

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
