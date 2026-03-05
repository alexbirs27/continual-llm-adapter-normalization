"""OLoRA: Orthogonal Low-Rank Adaptation for continual learning.

Implements dual LoRA matrices per target module:
  - lora_A / lora_B: frozen, accumulated from past tasks
  - loranew_A / loranew_B: trainable, current task

After each task, loranew is concatenated into lora and re-initialized.
An orthogonality loss encourages the new adapter subspace to be orthogonal
to the accumulated past subspace.
"""

import copy
import math

import torch
import torch.nn as nn


class OLoRALayer(nn.Module):
    """Replaces a single linear layer with dual LoRA decomposition."""

    def __init__(self, original_layer, r, alpha, dropout):
        super().__init__()
        self.original = original_layer
        for param in self.original.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.r = r
        self.scaling = alpha / r

        # Past task matrices (frozen) — None until after first task
        self.lora_A = None
        self.lora_B = None

        # Current task matrices (trainable)
        self.loranew_A = nn.Linear(in_features, r, bias=False)
        self.loranew_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize: A with Kaiming, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.loranew_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B.weight)

    def forward(self, x):
        result = self.original(x)

        # Past contribution (frozen)
        if self.lora_A is not None:
            past = self.lora_dropout(x)
            past = self.lora_A(past)
            past = self.lora_B(past)
            result = result + past * self.scaling

        # Current contribution (trainable)
        current = self.lora_dropout(x)
        current = self.loranew_A(current)
        current = self.loranew_B(current)
        result = result + current * self.scaling

        return result

    def concatenate_and_reinit(self):
        """After a task: merge loranew into lora, freeze, re-init loranew."""
        in_features = self.loranew_A.in_features
        out_features = self.loranew_B.out_features

        with torch.no_grad():
            if self.lora_A is None:
                # First task: move loranew -> lora
                self.lora_A = copy.deepcopy(self.loranew_A)
                self.lora_B = copy.deepcopy(self.loranew_B)
            else:
                # Concatenate: lora_A grows [r_sum, d] -> [r_sum+r, d]
                new_A_weight = torch.cat(
                    [self.lora_A.weight, self.loranew_A.weight], dim=0
                )
                new_B_weight = torch.cat(
                    [self.lora_B.weight, self.loranew_B.weight], dim=1
                )

                r_new = new_A_weight.shape[0]
                self.lora_A = nn.Linear(in_features, r_new, bias=False)
                self.lora_A.weight.copy_(new_A_weight)

                self.lora_B = nn.Linear(r_new, out_features, bias=False)
                self.lora_B.weight.copy_(new_B_weight)

            # Freeze past matrices
            self.lora_A.requires_grad_(False)
            self.lora_B.requires_grad_(False)

        # Re-init loranew for next task
        nn.init.kaiming_uniform_(self.loranew_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B.weight)


def _get_submodule(model, target_name):
    """Get a submodule by dot-separated path."""
    atoms = target_name.split(".")
    mod = model
    for atom in atoms:
        mod = getattr(mod, atom)
    return mod


def _set_submodule(model, target_name, new_module):
    """Set a submodule by dot-separated path."""
    atoms = target_name.split(".")
    parent = model
    for atom in atoms[:-1]:
        parent = getattr(parent, atom)
    setattr(parent, atoms[-1], new_module)


class OLoRA:
    """Orthogonal LoRA for continual learning.

    Injects OLoRALayer modules into the base model at target_modules.
    Manages the dual-matrix lifecycle across tasks.
    """

    def __init__(self, base_model, lora_config, lambda_1=0.5, lambda_2=0.0):
        self.model = base_model
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.olora_layers = {}

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Inject OLoRA layers
        self._inject_olora_layers(
            r=lora_config.r,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
        )

    def _inject_olora_layers(self, r, alpha, dropout, target_modules):
        """Replace target linear layers with OLoRALayer wrappers."""
        replacements = []
        for name, module in self.model.named_modules():
            # Check if this module's last name component matches a target
            short_name = name.split(".")[-1] if "." in name else name
            if short_name in target_modules and isinstance(module, nn.Linear):
                replacements.append((name, module))

        for name, module in replacements:
            olora_layer = OLoRALayer(module, r, alpha, dropout)
            olora_layer.to(module.weight.device)
            _set_submodule(self.model, name, olora_layer)
            self.olora_layers[name] = olora_layer

        print(f"Injected OLoRA into {len(self.olora_layers)} layers")

    def prepare_task(self, task_id):
        """No special prep needed — loranew is always ready."""
        pass

    def get_trainable_params(self):
        """Return only the loranew parameters (trainable)."""
        params = []
        for layer in self.olora_layers.values():
            params.extend(layer.loranew_A.parameters())
            params.extend(layer.loranew_B.parameters())
        return params

    def compute_orthogonal_loss(self):
        """L_orth = sum over layers of ||lora_A^T @ loranew_A||_1.

        Encourages the new adapter subspace to be orthogonal to past subspaces.
        lora_A.weight shape: [r_past, in_features]
        loranew_A.weight shape: [r, in_features]
        Product shape: [r_past, r]
        """
        loss = torch.tensor(0.0, device=next(iter(self.olora_layers.values())).loranew_A.weight.device)
        for layer in self.olora_layers.values():
            if layer.lora_A is not None:
                # [r_past, in_features] @ [in_features, r] -> [r_past, r]
                product = layer.lora_A.weight @ layer.loranew_A.weight.T
                loss = loss + torch.abs(product).sum()
        return loss

    def compute_l2_loss(self):
        """L2 regularization on loranew parameters."""
        loss = torch.tensor(0.0, device=next(iter(self.olora_layers.values())).loranew_A.weight.device)
        for layer in self.olora_layers.values():
            loss = loss + torch.norm(layer.loranew_A.weight, p=2)
            loss = loss + torch.norm(layer.loranew_B.weight, p=2)
        return loss

    def get_loss(self, input_ids, attention_mask, labels):
        """CE loss + orthogonality loss + L2 regularization."""
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
        """Concatenate loranew into lora, freeze, re-init loranew."""
        for layer in self.olora_layers.values():
            layer.concatenate_and_reinit()

    def set_eval_adapter(self, task_id):
        """OLoRA uses a single accumulated model — no adapter switching needed."""
        pass

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
