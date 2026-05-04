"""Extract per-task LoRA A and B matrices from trained OLoRA and IncLoRA methods."""

import re
import torch


_PARAM_SUFFIX = re.compile(r'\.parametrizations\.weight\.\d+$')
_LAYER_IDX    = re.compile(r'\.layers\.(\d+)\.')
_MODULE_TYPE  = re.compile(r'\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\b')


def _clean_name(name: str) -> str:
    """Strip the parametrization suffix so OLoRA and IncLoRA layer names match."""
    return _PARAM_SUFFIX.sub('', name)


def extract_olora_adapters(method, r: int, num_tasks: int,
                           target_modules=('q_proj', 'v_proj')):
    """
    Extract per-task A / B matrices from OLoRA's concatenated lora_A / lora_B.

    OLoRA stores past tasks by concatenating along the rank dim after each task:
      lora_A shape  →  (num_tasks * r,  fan_in)
      lora_B shape  →  (fan_out,        num_tasks * r)

    Task t's slice:
      A_t = lora_A[t*r : (t+1)*r, :]          shape (r, fan_in)
      B_t = lora_B[:,   t*r : (t+1)*r]         shape (fan_out, r)

    Returns:
        {task_id: {layer_name: {'A': Tensor, 'B': Tensor}}}
    """
    from src.methods.olora import OLoRAParametrization

    adapters = {t: {} for t in range(num_tasks)}

    # Unwrap DDP if needed
    base = getattr(method.model, 'module', method.model)

    for raw_name, module in base.named_modules():
        if not isinstance(module, OLoRAParametrization):
            continue
        if not any(m in raw_name for m in target_modules):
            continue
        if not module.has_past:
            continue

        layer_name = _clean_name(raw_name)
        lora_A = module.lora_A.detach().cpu().float()  # (num_tasks*r, fan_in)
        lora_B = module.lora_B.detach().cpu().float()  # (fan_out, num_tasks*r)

        actual_tasks = lora_A.shape[0] // r
        for t in range(min(num_tasks, actual_tasks)):
            adapters[t][layer_name] = {
                'A': lora_A[t * r:(t + 1) * r, :],
                'B': lora_B[:, t * r:(t + 1) * r],
            }

    return adapters


def extract_inclora_adapters(method, target_modules=('q_proj', 'v_proj')):
    """
    Extract per-task A / B matrices from IncLoRA's saved state dicts.

    IncLoRA saves one state dict per task in method.saved_lora_states.
    Keys look like:
      model.layers.0.self_attn.q_proj.parametrizations.weight.0.lora_A

    Returns:
        {task_id: {layer_name: {'A': Tensor, 'B': Tensor}}}
    """
    adapters = {}
    key_re = re.compile(r'^(.+)\.parametrizations\.weight\.\d+\.(lora_[AB])$')

    for task_id, state_dict in enumerate(method.saved_lora_states):
        layer_data = {}

        for key, tensor in state_dict.items():
            if not any(m in key for m in target_modules):
                continue
            match = key_re.match(key)
            if not match:
                continue
            layer_name = match.group(1)
            mat_key = 'A' if match.group(2) == 'lora_A' else 'B'
            if layer_name not in layer_data:
                layer_data[layer_name] = {}
            layer_data[layer_name][mat_key] = tensor.detach().cpu().float()

        adapters[task_id] = layer_data

    return adapters


def delta_W(entry: dict) -> torch.Tensor:
    """Compute the low-rank weight update  B @ A  →  (fan_out, fan_in)."""
    return entry['B'] @ entry['A']


def save_adapters(adapters: dict, path: str):
    torch.save(adapters, path)


def load_adapters(path: str) -> dict:
    return torch.load(path, map_location='cpu')


def layer_index(layer_name: str) -> int:
    m = _LAYER_IDX.search(layer_name)
    return int(m.group(1)) if m else -1


def module_type(layer_name: str) -> str:
    m = _MODULE_TYPE.search(layer_name)
    return m.group(1) if m else 'unknown'
