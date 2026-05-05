import torch
import torch.nn.functional as F


def adversarial_loss(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    """LSGAN loss: MSE against all-ones (real) or all-zeros (fake) targets."""
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def feature_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.2, delta: float = 1.0
) -> torch.Tensor:
    """Combined Huber + cosine loss for feature vectors.

    alpha=1.0  pure Hube
    alpha=0.0  pure (1 - cosine_similarity)
    0 < alpha < 1  linear blend
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    huber = F.huber_loss(pred, target, delta=delta)
    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean()

    return alpha * huber + (1 - alpha) * cos


def cycle_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    alpha: float = 1.0,
    delta: float = 1.0,
) -> torch.Tensor:
    return feature_loss(reconstructed, original, alpha)


def identity_loss(
    translated: torch.Tensor,
    original: torch.Tensor,
    alpha: float = 1.0,
    delta: float = 1.0,
) -> torch.Tensor:
    return feature_loss(translated, original, alpha)


def paired_loss(
    translated: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    delta: float = 1.0,
) -> torch.Tensor:
    return feature_loss(
        translated, target, 0.0, delta
    )  # Paired loss is pure cosine similarity


def coral_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    n = x.shape[0]

    if n < x.shape[1]:
        # batch < feature_dim: use the Gram matrix [n, n] instead of the
        # feature covariance [d, d]. Rank is identical (min(n, d) = n in both
        # cases) but the Gram avoids the (d-n)^2 zero block that dominates the
        # MSE and dilutes gradients when d >> n.
        cx = x @ x.T / (n - 1)
        cy = y @ y.T / (n - 1)
    else:
        cx = x.T @ x / (n - 1)
        cy = y.T @ y / (n - 1)

    return F.mse_loss(cx, cy)


def relational_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sx = F.normalize(x, dim=-1) @ F.normalize(x, dim=-1).T
    sy = F.normalize(y, dim=-1) @ F.normalize(y, dim=-1).T
    return F.mse_loss(sx, sy)
