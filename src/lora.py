"""
LoRA (Hu et al., 2021), written out rather than imported so every piece is visible.

A frozen weight W gets a trainable low-rank update:  W x  ->  W x + (alpha / r) * B A x
with A: (r, in) and B: (out, r). B starts at zero, so training begins exactly at the
pretrained model; only A and B (and the classifier head) receive gradients.
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))    # same init as nn.Linear
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.T @ self.B.T) * self.scale


def inject(model: nn.Module, r: int, alpha: float, targets=("query", "value")):
    """Freeze the model, then wrap every Linear whose name ends in one of `targets`."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha))
                n += 1
    if n == 0:
        raise ValueError(f"no Linear layers named {targets} found")
    return n


def count(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
