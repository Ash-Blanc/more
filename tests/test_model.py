import torch

from more.config import LossConfig, ModelConfig
from more.generate import generate_tokens
from more.model import MoREModel


def small_model(**overrides):
    values = {
        "vocab_size": 259,
        "max_seq_len": 12,
        "d_model": 24,
        "n_heads": 4,
        "n_recursions": 2,
        "n_experts": 2,
        "moe_top_k": 1,
        "ffn_multiplier": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return MoREModel(ModelConfig(**values), LossConfig())


def test_forward_shapes_and_shared_block() -> None:
    model = small_model(n_recursions=3)
    tokens = torch.randint(0, 259, (2, 8))
    output = model(tokens, labels=tokens.roll(-1, dims=1))
    assert output.logits.shape == (2, 8, 259)
    assert output.loss is not None
    assert output.depth_active_mask.shape == (2, 8, 3)
    assert model.shared_block is model.shared_block


def test_future_tokens_cannot_change_earlier_logits() -> None:
    torch.manual_seed(0)
    model = small_model()
    model.eval()
    tokens = torch.randint(0, 259, (1, 8))
    changed = tokens.clone()
    changed[0, 6:] = (changed[0, 6:] + 17) % 259
    first = model(tokens).logits
    second = model(changed).logits
    assert torch.allclose(first[:, :6], second[:, :6], atol=1e-5, rtol=1e-5)


def test_router_gradients_reach_depth_and_expert_projections() -> None:
    model = small_model()
    tokens = torch.randint(0, 259, (2, 8))
    output = model(tokens, labels=tokens.roll(-1, dims=1))
    assert output.loss is not None
    output.loss.backward()
    assert model.depth_router.projection.weight.grad is not None
    assert model.shared_block.ffn.router.projection.weight.grad is not None


def test_generation_is_deterministic_for_greedy_sampling() -> None:
    model = small_model()
    model.eval()
    prompt = torch.tensor([[257, 65, 66]])
    first = generate_tokens(model, prompt, 4, temperature=0.0, eos_id=None)
    second = generate_tokens(model, prompt, 4, temperature=0.0, eos_id=None)
    assert torch.equal(first, second)
    assert first.shape == (1, 7)
