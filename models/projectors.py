"""Feature-space transition projectors used by RSIAT variants."""

import math

import torch
from torch import nn

from utils.toolkit import AutoencoderSigmoid


class IdentityProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_zero", torch.zeros(()))

    def forward(self, features):
        return features

    def regularization_loss(self):
        return self._zero


class LowRankResidualProjector(nn.Module):
    """Identity plus a low-rank linear transition."""

    def __init__(self, input_dim=768, rank=16):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, input_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return features + self.up(self.down(features))

    def regularization_loss(self):
        return self.up.weight.pow(2).mean()


class WeaklyNonlinearProjector(nn.Module):
    """Identity + gated low-rank linear drift + a small nonlinear residual.

    Both residual branches end in zero-initialized layers, so the initial
    transition is exactly the identity. This is important when a projector is
    fitted only on the current task and later evaluated around old prototypes.
    """

    def __init__(
        self,
        input_dim=768,
        rank=16,
        hidden_dim=64,
        dropout=0.0,
        linear_gate_init=0.5,
        nonlinear_gate_init=0.1,
    ):
        super().__init__()
        self.linear_down = nn.Linear(input_dim, rank, bias=False)
        self.linear_up = nn.Linear(rank, input_dim, bias=False)
        self.nonlinear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        nn.init.kaiming_uniform_(self.linear_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_up.weight)
        nn.init.zeros_(self.nonlinear[-1].weight)
        nn.init.zeros_(self.nonlinear[-1].bias)

        self.linear_gate_logit = nn.Parameter(
            torch.tensor(self._to_logit(linear_gate_init), dtype=torch.float32)
        )
        self.nonlinear_gate_logit = nn.Parameter(
            torch.tensor(self._to_logit(nonlinear_gate_init), dtype=torch.float32)
        )

    @staticmethod
    def _to_logit(value):
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        return math.log(value / (1.0 - value))

    @property
    def linear_gate(self):
        return torch.sigmoid(self.linear_gate_logit)

    @property
    def nonlinear_gate(self):
        return torch.sigmoid(self.nonlinear_gate_logit)

    def residual_components(self, features):
        linear = self.linear_up(self.linear_down(features))
        nonlinear = self.nonlinear(features)
        return self.linear_gate * linear, self.nonlinear_gate * nonlinear

    def forward(self, features):
        linear, nonlinear = self.residual_components(features)
        return features + linear + nonlinear

    def regularization_loss(self):
        # Penalize nonlinear capacity more strongly; the caller controls the
        # global weight of this term through the experiment config.
        nonlinear_weight = self.nonlinear[-1].weight.pow(2).mean()
        return self.nonlinear_gate.pow(2) + nonlinear_weight


def build_projector(args, input_dim=768):
    name = args.get("projector_type", "rae").lower()
    if name == "identity":
        return IdentityProjector()
    if name in {"linear", "low_rank", "low_rank_residual"}:
        return LowRankResidualProjector(
            input_dim=input_dim,
            rank=args.get("projector_rank", 16),
        )
    if name in {"weakly_nonlinear", "weak_nonlinear", "wln"}:
        return WeaklyNonlinearProjector(
            input_dim=input_dim,
            rank=args.get("projector_rank", 16),
            hidden_dim=args.get("projector_hidden_dim", 64),
            dropout=args.get("projector_dropout", 0.0),
            linear_gate_init=args.get("linear_gate_init", 0.5),
            nonlinear_gate_init=args.get("nonlinear_gate_init", 0.1),
        )
    if name in {"rae", "residual_autoencoder"}:
        return AutoencoderSigmoid(
            input_dims=input_dim,
            code_dims=args.get("ae_code_dims", 384),
        )
    raise ValueError("Unknown projector_type: {}".format(name))
