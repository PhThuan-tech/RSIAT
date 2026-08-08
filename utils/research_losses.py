"""Losses for moment-aware feature transport and adaptive separation."""

import torch
from torch.nn import functional as F


def _moments(features):
    mean = features.mean(dim=0)
    variance = features.var(dim=0, unbiased=False)
    return mean, variance


def moment_alignment_loss(current, projected, labels=None, classwise=True):
    """Match first and diagonal second moments.

    Class-wise estimates are used when a mini-batch contains at least two
    examples from a class. If no class has enough examples, task-level moments
    provide a stable fallback.
    """
    pairs = []
    if classwise and labels is not None:
        for label in torch.unique(labels):
            mask = labels == label
            if int(mask.sum()) >= 2:
                pairs.append((current[mask], projected[mask]))
    if not pairs:
        pairs = [(current, projected)]

    mean_losses = []
    variance_losses = []
    for current_group, projected_group in pairs:
        current_mean, current_variance = _moments(current_group)
        projected_mean, projected_variance = _moments(projected_group)
        mean_losses.append(F.mse_loss(projected_mean, current_mean))
        variance_losses.append(F.mse_loss(projected_variance, current_variance))

    return torch.stack(mean_losses).mean(), torch.stack(variance_losses).mean()


def adaptive_topk_separation_loss(
    current_features,
    old_prototypes,
    topk=10,
    threshold_min=0.1,
    threshold_max=0.5,
    reference_features=None,
    reference_prototypes=None,
):
    """Separate new features from only their most confusing old prototypes.

    The threshold is higher for semantically/representationally close pairs,
    allowing related classes to share structure instead of forcing universal
    orthogonality. Returns both the loss and lightweight diagnostics.
    """
    if old_prototypes is None or old_prototypes.numel() == 0:
        zero = current_features.sum() * 0.0
        return zero, {"topk_similarity": zero.detach(), "active_fraction": zero.detach()}

    current = F.normalize(current_features, p=2, dim=1)
    prototypes = F.normalize(old_prototypes, p=2, dim=1)
    similarities = current @ prototypes.t()
    k = min(int(topk), similarities.shape[1])
    top_values, top_indices = torch.topk(similarities, k=k, dim=1)

    if reference_features is not None and reference_prototypes is not None:
        reference = F.normalize(reference_features.detach(), p=2, dim=1)
        reference_proto = F.normalize(reference_prototypes.detach(), p=2, dim=1)
        relatedness = (reference @ reference_proto.t()).gather(1, top_indices)
        relatedness = ((relatedness + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        relatedness = ((top_values.detach() + 1.0) * 0.5).clamp(0.0, 1.0)

    thresholds = threshold_min + (threshold_max - threshold_min) * relatedness
    violations = F.relu(top_values - thresholds)
    loss = violations.mean()
    diagnostics = {
        "topk_similarity": top_values.detach().mean(),
        "active_fraction": (violations.detach() > 0).float().mean(),
    }
    return loss, diagnostics
