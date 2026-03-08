import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import math
from functools import partial
from src.minlora.model import LoRAParametrization

class ELLAParametrization(LoRAParametrization):
    """Extends minLoRA for ELLA: Efficient Lifelong Learning for Adapters."""

    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=8, lora_dropout_p=0.0, lora_alpha=1):
        # Initialize basic LoRA structures (frozen base is handled by parametrize)
        nn.Module.__init__(self)
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # W_past: The cumulative signal of all past task updates (frozen)
        self.register_buffer("W_past", torch.zeros(self.swap((fan_out, fan_in))))

        # Current task matrices (trainable)
        self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))))
        self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))))
                
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype))

    def _dropout(self, A):
        return A * self.lora_dropout(self.lora_dropout_mask)

    def get_delta_w(self):
        """Returns the current low-rank update matrix Delta W = AB"""
        return torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A)))) * self.scaling

    def forward(self, X):        
        # Base + Past Tasks (Frozen) + Current Task (Trainable)
        delta_w_current = self.get_delta_w()
        total_delta_w = self.W_past + delta_w_current
        return X + (X @ total_delta_w.t()).view(X.shape)

    @torch.no_grad()
    def update_past_signal(self):
        """Accumulate current update into W_past and reset for next task"""
        current_delta = self.get_delta_w()
        self.W_past.add_(current_delta)
        
        # Re-initialize for the new task to maintain plasticity
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )
    
    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )
    

def add_ella_by_name(model, target_module_names, ella_config):
    """Add ELLA parametrization to specific layers, same pattern as minLoRA's add_lora_by_name."""
    for name, layer in model.named_modules():
        if any(m in name for m in target_module_names):
            if type(layer) in ella_config:
                for attr_name, parametrization_fn in ella_config[type(layer)].items():
                    parametrize.register_parametrization(layer, attr_name, parametrization_fn(layer))

class ELLA:    

    def __init__(self, base_model, lora_config, lambd=0.1):
        self.model = base_model
        self.lambd = lambd # Regularization strength

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Inject ELLA parametrizations (same pattern as minLoRA)
        ella_config = {
            nn.Linear: {
                "weight": partial(
                    ELLAParametrization.from_linear,
                    rank=lora_config.r,
                    lora_dropout_p=lora_config.dropout,
                    lora_alpha=lora_config.alpha,
                ),
            },
            nn.Embedding: {
                "weight": partial(
                    ELLAParametrization.from_embedding,
                    rank=lora_config.r,
                    lora_dropout_p=lora_config.dropout,
                    lora_alpha=lora_config.alpha,
                ),
            },
        }
        add_ella_by_name(self.model, list(lora_config.target_modules), ella_config)

        # Collect all ELLA layers
        self.ella_layers = [m for m in self.model.modules() if isinstance(m, ELLAParametrization)]
        print(f"Injected ELLA into {len(self.ella_layers)} layers")

    def get_trainable_params(self):
        params = []
        for layer in self.ella_layers:
            params.append(layer.lora_A)
            params.append(layer.lora_B)
        return params

    def compute_ella_loss(self):
        """
        Calculates the alignment penalty to discourage overlap in high-energy 
        task-specific directions.
        """
        loss = 0.0
        for layer in self.ella_layers:
            delta_w = layer.get_delta_w()
            # Element-wise product of current update and past energy
            alignment = delta_w * layer.W_past 
            loss += torch.norm(alignment, p='fro')**2
        return loss

    def get_loss(self, input_ids, attention_mask, labels):
        """Total loss = Task Accuracy Loss + Lambda * ELLA Penalty"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        task_loss = outputs.loss
        
        ella_penalty = self.compute_ella_loss()
        return task_loss + self.lambd * ella_penalty

    def after_task(self):        
        for layer in self.ella_layers:
            layer.update_past_signal()
    
    def set_eval_adapter(self, task_id):
        pass

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
