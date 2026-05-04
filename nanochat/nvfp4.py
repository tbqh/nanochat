"""NVFP4 training integration for nanochat using TorchAO."""

import torch.nn as nn


class _TempLinearContainer(nn.Module):
    """Temporary container to allow quantize_ to work on a single Linear."""
    def __init__(self, linear):
        super().__init__()
        self.linear = linear


def convert_to_nvfp4_training(model, module_filter_fn=None):
    """Apply NVFP4 quantization to model's Linear layers.

    Args:
        model: The model to quantize
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching
            Linears are converted. Use to skip layers with incompatible dims.

    Returns:
        The quantized model (modified in-place), number of quantized layers
    """
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats.nvfp4_training import NVFP4TrainingConfig

    config = NVFP4TrainingConfig()
    num_quantized = 0

    # Collect linear layers and their parent modules
    modules_to_quantize = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if module_filter_fn is None or module_filter_fn(mod, name):
                # Find parent module
                if '.' in name:
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                modules_to_quantize.append((parent, child_name, mod))

    # Quantize each module individually using a temp container
    for parent, child_name, mod in modules_to_quantize:
        container = _TempLinearContainer(mod)
        quantize_(container, config)
        setattr(parent, child_name, container.linear)
        num_quantized += 1

    return model, num_quantized


def nvfp4_module_filter(mod: nn.Module, fqn: str) -> bool:
    """Default filter for NVFP4: skip incompatible layers.

    Skips: embeddings, lm_head (per NVIDIA paper Section 4.1), and layers
    with dimensions not divisible by 16 (block quantization requirement).
    """
    if not isinstance(mod, nn.Linear):
        return False
    # Skip lm_head (output logits) - quantizing this can cause numerical issues
    if "lm_head" in fqn:
        return False
    # Skip embedding-related layers
    if "wte" in fqn or "embed" in fqn:
        return False
    # Skip if dimensions not divisible by 16
    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
        return False
    return True
