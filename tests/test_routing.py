import torch

from more.routing import (
    DepthRouter,
    TopKRouter,
    hierarchical_active_mask,
    load_balance_loss,
    router_z_loss,
)


def test_hierarchical_mask_is_prefix_closed() -> None:
    assignments = torch.tensor([[0, 1, 2]])
    mask = hierarchical_active_mask(assignments, 3)
    assert mask.tolist() == [[[True, False, False], [True, True, False], [True, True, True]]]


def test_top_k_gates_are_sparse_and_normalized() -> None:
    router = TopKRouter(d_model=8, n_experts=4, top_k=2)
    result = router(torch.randn(2, 3, 8))
    assert (result.gates > 0).sum(dim=-1).eq(2).all()
    assert torch.allclose(result.gates.sum(dim=-1), torch.ones(2, 3))


def test_router_losses_are_finite() -> None:
    logits = torch.randn(2, 4, 3)
    probs = logits.softmax(-1)
    assignments = probs.argmax(-1)
    assert torch.isfinite(load_balance_loss(probs, assignments))
    assert torch.isfinite(router_z_loss(logits))


def test_depth_router_exposes_trainable_straight_through_gate() -> None:
    router = DepthRouter(d_model=8, n_recursions=3)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    result = router(hidden)
    result.straight_through_active.sum().backward()
    assert router.projection.weight.grad is not None
