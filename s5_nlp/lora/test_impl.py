import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from s5_nlp.gpt.impl import GPT, GPTConfig, estimate_loss

from .impl import LoRALinear, apply_lora, count_parameters, finetune_lm, merge_lora


def test_starts_as_the_base_layer_and_counts_parameters():
    torch.manual_seed(0)
    base = nn.Linear(64, 32)
    lora = LoRALinear(copy.deepcopy(base), r=4)
    x = torch.randn(5, 64)
    torch.testing.assert_close(lora(x), base(x))  # B = 0 at init
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert sorted(trainable) == ["A", "B"]
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) == 4 * (64 + 32)


def test_forward_equals_explicit_low_rank_update_and_merging():
    torch.manual_seed(1)
    lora = LoRALinear(nn.Linear(10, 6), r=2, alpha=4)
    nn.init.normal_(lora.B)
    x = torch.randn(3, 10)
    W_eff = lora.base.weight + (4 / 2) * lora.B @ lora.A
    torch.testing.assert_close(lora(x), x @ W_eff.T + lora.base.bias)
    assert torch.linalg.matrix_rank(lora.delta_weight()) <= 2
    before = lora(x)
    lora.merge()
    torch.testing.assert_close(lora(x), before)  # same function, no extra matmuls
    lora.unmerge()
    torch.testing.assert_close(lora(x), before)


def test_only_adapters_receive_gradients():
    torch.manual_seed(2)
    lora = LoRALinear(nn.Linear(4, 3), r=2)
    nn.init.normal_(lora.B)
    lora(torch.randn(5, 4)).sum().backward()
    assert lora.base.weight.grad is None and lora.A.grad is not None and lora.B.grad is not None
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    lora64 = lora.double()
    assert torch.autograd.gradcheck(lambda t: lora64(t), (x,))


def test_apply_to_gpt_wraps_attention_layers_only():
    torch.manual_seed(3)
    cfg = GPTConfig(vocab_size=20, block_size=32, n_layer=2, n_head=2, d_model=32, dropout=0.0)
    base = GPT(cfg).eval()
    model = apply_lora(copy.deepcopy(base), r=4)
    wrapped = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    assert sorted(wrapped) == sorted(
        [f"blocks.{i}.attn.{n}" for i in range(2) for n in ("qkv", "proj")]
    )
    idx = torch.randint(0, 20, (2, 10))
    torch.testing.assert_close(model(idx)[0], base(idx)[0])
    trainable, total = count_parameters(model)
    assert trainable == 2 * (4 * (32 + 96) + 4 * (32 + 32)) and trainable < 0.1 * total


@pytest.mark.slow
def test_lora_adapts_the_shakespeare_gpt_to_a_new_domain():
    from s5_nlp.bert_classifier.impl import make_dataset
    from s5_nlp.gpt.impl import pretrained_shakespeare_gpt

    base, tok, _ = pretrained_shakespeare_gpt()
    reviews, _ = make_dataset(3000, seed=0)  # lowercase movie reviews: a new domain
    text = ". ".join(reviews) + "."
    data = torch.tensor(tok.encode(text))
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]
    before = estimate_loss(base, val, 64)

    lora = apply_lora(copy.deepcopy(base), r=4, alpha=8)
    lora_trainable, total = count_parameters(lora)
    lora_hist = finetune_lm(lora, train, steps=300, lr=3e-3)
    merge_lora(lora)
    after_lora = estimate_loss(lora, val, 64)

    full = copy.deepcopy(base)
    full_hist = finetune_lm(full, train, steps=300, lr=1e-3)
    after_full = estimate_loss(full, val, 64)

    assert lora_trainable < 0.05 * total
    assert after_lora < 0.5 * before  # large gain from ~3% of the parameters...
    assert after_full <= after_lora * 1.2  # ...close to updating everything

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(lora_hist, label=f"LoRA r=4 ({lora_trainable:,} params) val {after_lora:.3f}")
    ax.plot(full_hist, label=f"full fine-tune ({total - lora_trainable:,}) val {after_full:.3f}")
    ax.axhline(before, ls="--", c="gray", label=f"before adaptation {before:.3f}")
    ax.set(xlabel="step", ylabel="nats / char", title="Shakespeare GPT -> movie reviews")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    print(
        f"before {before:.3f} lora {after_lora:.3f} full {after_full:.3f} params {lora_trainable}/{total}"
    )
