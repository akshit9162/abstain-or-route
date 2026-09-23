import pytest
import torch
import torch.nn as nn
from src.lora import LoRALinear, inject, count


class Block(nn.Module):
    """Stand-in for an attention block: the same child names inject() looks for."""
    def __init__(self, d=16):
        super().__init__()
        self.query, self.key, self.value = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return self.query(x) + self.key(x) + self.value(x)


def toy(n_blocks=2, d=16):
    torch.manual_seed(0)
    return nn.Sequential(*[Block(d) for _ in range(n_blocks)])


def test_starts_exactly_at_the_pretrained_model():
    m = toy()
    x = torch.randn(4, 16)
    before = m(x)
    inject(m, r=4, alpha=8)
    m.eval()                                   # no dropout, so B = 0 must mean an identical output
    assert torch.allclose(m(x), before)


def test_only_low_rank_factors_train():
    m = toy(n_blocks=3, d=16)
    n = inject(m, r=4, alpha=8)
    assert n == 6                              # query + value in each of 3 blocks; key untouched
    trainable = {name for name, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(name.endswith((".A", ".B")) for name in trainable)
    t, _ = count(m)
    assert t == 6 * 4 * (16 + 16)              # r * (in + out) per wrapped layer


def test_update_is_scaled_low_rank_product():
    base = nn.Linear(8, 6)
    layer = LoRALinear(base, r=2, alpha=6, dropout=0.0)
    with torch.no_grad():
        layer.B.normal_()
    x = torch.randn(3, 8)
    expected = base(x) + (6 / 2) * x @ (layer.B @ layer.A).T
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_gradients_reach_A_and_B_but_not_the_frozen_weight():
    layer = LoRALinear(nn.Linear(8, 6), r=2, alpha=4, dropout=0.0)
    with torch.no_grad():
        layer.B.normal_()                      # with B = 0, A's gradient would be zero on step one
    layer(torch.randn(5, 8)).sum().backward()
    assert layer.A.grad is not None and layer.A.grad.abs().sum() > 0
    assert layer.B.grad is not None and layer.B.grad.abs().sum() > 0
    assert layer.base.weight.grad is None


def test_inject_refuses_a_model_with_no_targets():
    with pytest.raises(ValueError):
        inject(nn.Sequential(nn.Linear(4, 4)), r=2, alpha=4)
