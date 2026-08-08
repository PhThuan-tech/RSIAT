import unittest

import torch

from models.projectors import WeaklyNonlinearProjector
from utils.research_losses import (
    adaptive_topk_separation_loss,
    moment_alignment_loss,
)


class WeaklyNonlinearProjectorTests(unittest.TestCase):
    def test_zero_initialized_projector_is_identity(self):
        projector = WeaklyNonlinearProjector(input_dim=8, rank=2, hidden_dim=4)
        features = torch.randn(5, 8)
        self.assertTrue(torch.allclose(projector(features), features, atol=1e-7))

    def test_projector_receives_gradients(self):
        projector = WeaklyNonlinearProjector(input_dim=8, rank=2, hidden_dim=4)
        features = torch.randn(5, 8)
        target = torch.randn(5, 8)
        loss = (projector(features) - target).pow(2).mean()
        loss.backward()
        self.assertIsNotNone(projector.linear_up.weight.grad)
        self.assertIsNotNone(projector.nonlinear[-1].weight.grad)


class ResearchLossTests(unittest.TestCase):
    def test_identical_features_have_zero_moment_loss(self):
        features = torch.randn(6, 8)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        mean_loss, variance_loss = moment_alignment_loss(
            features,
            features.clone(),
            labels=labels,
        )
        self.assertAlmostEqual(mean_loss.item(), 0.0, places=7)
        self.assertAlmostEqual(variance_loss.item(), 0.0, places=7)

    def test_topk_separation_is_finite_and_differentiable(self):
        features = torch.randn(4, 8, requires_grad=True)
        prototypes = torch.randn(3, 8)
        loss, diagnostics = adaptive_topk_separation_loss(
            features,
            prototypes,
            topk=2,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("active_fraction", diagnostics)
        loss.backward()
        self.assertIsNotNone(features.grad)


if __name__ == "__main__":
    unittest.main()
