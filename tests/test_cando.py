"""Sanity tests for the cando package (CPU only).  Run:  python -m pytest tests"""
import torch

from cando import cando, max_prob_confidence, rodrigues, rotate


def test_rodrigues_is_rotation():
    R = rodrigues(torch.randn(16, 3))
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(16), atol=1e-5)
    assert torch.allclose(rodrigues(torch.zeros(3)), torch.eye(3))


def test_max_prob_confidence_shapes():
    assert max_prob_confidence(torch.rand(5, 40).softmax(-1)).shape == (5,)
    assert max_prob_confidence(torch.rand(5, 100, 4).softmax(-1)).shape == (5,)


def test_recovers_known_rotation():
    target = rodrigues(torch.tensor([0.4, -1.1, 0.7]))           # objective peaks at R = target

    def objective(Rs):
        return -(Rs - target).flatten(1).norm(dim=-1)

    R, _, info = cando(objective, budget=300, device="cpu")
    assert (R - target).norm() < 0.05
    assert info["queries"] <= 300


def test_identity_kept_with_prior():
    def objective(Rs):                                            # flat landscape: nothing to gain
        return torch.zeros(Rs.shape[0])

    R, theta, _ = cando(objective, budget=50, lam=0.1, device="cpu")
    assert torch.allclose(R, torch.eye(3)) and theta.norm() == 0


def test_rotate_batch_matches_single():
    P, Rs = torch.randn(64, 3), rodrigues(torch.randn(4, 3))
    assert torch.allclose(rotate(P, Rs)[2], rotate(P, Rs[2]), atol=1e-6)
