"""OLoRA: Orthogonal Low-Rank Adaptation for continual learning.

Extends minLoRA's LoRAParametrization (copied into src/minlora) with dual matrices:
  - lora_A / lora_B: frozen, accumulated from past tasks
  - loranew_A / loranew_B: trainable, current task

Uses the exact same torch.nn.utils.parametrize mechanism as minLoRA.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from src.minlora.model import LoRAParametrization


class OLoRAParametrization(LoRAParametrization):
    """Extends minLoRA's LoRAParametrization with dual matrices for O-LoRA."""

    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        # Don't call super().__init__ — we set everything up ourselves
        nn.Module.__init__(self)
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # Past task matrices (frozen) — empty until after first task
        self.has_past = False
        self.lora_A = nn.Parameter(torch.empty(0), requires_grad=False)
        self.lora_B = nn.Parameter(torch.empty(0), requires_grad=False)

        # Current task matrices (trainable) — same init as minLoRA
        self.loranew_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))))
        self.loranew_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))))
        nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.loranew_A.dtype))

    def _dropout(self, A):
        return A * self.lora_dropout(self.lora_dropout_mask)

    def forward(self, X):
        # Past contribution (frozen)
        if self.has_past:
            X = X + torch.matmul(*self.swap((self.lora_B, self.lora_A))).view(X.shape) * self.scaling
        # Current contribution (trainable)
        X = X + torch.matmul(*self.swap((self.loranew_B, self.dropout_fn(self.loranew_A)))).view(X.shape) * self.scaling
        return X

    def concatenate_and_reinit(self):
        """After a task: merge loranew into lora, freeze, re-init loranew."""
        with torch.no_grad():
            if not self.has_past:
                new_A = self.loranew_A.data.clone()
                new_B = self.loranew_B.data.clone()
            else:
                # Concat along rank dimension
                new_A = torch.cat([self.lora_A.data, self.loranew_A.data], dim=0)
                new_B = torch.cat([self.lora_B.data, self.loranew_B.data], dim=1)

            self.lora_A = nn.Parameter(new_A, requires_grad=False)
            self.lora_B = nn.Parameter(new_B, requires_grad=False)
            self.has_past = True

        # Re-init loranew for next task
        nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B)

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )


def add_olora_by_name(model, target_module_names, olora_config):
    """Add OLoRA parametrization to specific layers, same pattern as minLoRA's add_lora_by_name."""
    for name, layer in model.named_modules():
        if any(m in name for m in target_module_names):
            if type(layer) in olora_config:
                for attr_name, parametrization_fn in olora_config[type(layer)].items():
                    parametrize.register_parametrization(layer, attr_name, parametrization_fn(layer))


class OLoRA:
    """Orthogonal LoRA for continual learning."""

    def __init__(self, base_model, lora_config, lambda_1=0.5, lambda_2=0.0):
        self.model = base_model
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Inject OLoRA parametrizations (same pattern as minLoRA)
        olora_config = {
            nn.Linear: {
                "weight": partial(
                    OLoRAParametrization.from_linear,
                    rank=lora_config.r,
                    lora_dropout_p=lora_config.dropout,
                    lora_alpha=lora_config.alpha,
                ),
            },
        }
        add_olora_by_name(self.model, list(lora_config.target_modules), olora_config)

        # Collect all OLoRA layers
        self.olora_layers = [m for m in self.model.modules() if isinstance(m, OLoRAParametrization)]
        print(f"Injected OLoRA into {len(self.olora_layers)} layers")

    def prepare_task(self, task_id):
        pass

    def get_trainable_params(self):
        params = []
        for layer in self.olora_layers:
            params.append(layer.loranew_A)
            params.append(layer.loranew_B)
        return params

    def compute_orthogonal_loss(self):
        loss = torch.tensor(0.0, device=self.olora_layers[0].loranew_A.device)
        for layer in self.olora_layers:
            if layer.has_past:
                product = layer.lora_A @ layer.loranew_A.T
                loss = loss + torch.abs(product).sum()
        return loss

    def compute_l2_loss(self):
        loss = torch.tensor(0.0, device=self.olora_layers[0].loranew_A.device)
        for layer in self.olora_layers:
            loss = loss + torch.norm(layer.loranew_A, p=2)
            loss = loss + torch.norm(layer.loranew_B, p=2)
        return loss

    def get_loss(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss = outputs.loss
        if self.lambda_1 > 0:
            total_loss = total_loss + self.lambda_1 * self.compute_orthogonal_loss()
        if self.lambda_2 > 0:
            total_loss = total_loss + self.lambda_2 * self.compute_l2_loss()
        return total_loss

    def after_task(self, task_id):
        for layer in self.olora_layers:
            layer.concatenate_and_reinit()

    def set_eval_adapter(self, task_id):
        pass

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
