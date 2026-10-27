"""Training objectives; independent evaluation estimators live in metrics.py."""
import torch
from torch import nn
from . import upstream


def corr(x, y, eps=1e-9):
    x, y = x.reshape(-1), y.reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    # Mirrors Grari's sample-standard-deviation normalization and batch-mean product.
    return (x * y).mean() / ((x.var(unbiased=True) + eps) * (y.var(unbiased=True) + eps)).sqrt()


def penalty(method, p, s, degree=3):
    if method == "baseline":
        return p.sum() * 0
    if method == "pearson":
        # Population-moment Pearson squared (zero variance maps continuously to zero).
        pc, sc = p - p.mean(), s - s.mean()
        return (pc * sc).mean().square() / (pc.square().mean() * sc.square().mean() + 1e-12)
    if method == "mary_chi2":
        # No surrogate: original KDE, grid, bandwidth and chi-square formula.
        # Constant predictions make the original normalization undefined: explicit guard.
        if p.std().detach() < 1e-7 or s.std().detach() < 1e-7:
            return p.sum() * 0
        if p.device.type != "cpu":
            raise ValueError("Original Mary implementation creates CPU grids; use CPU.")
        return upstream.mary.chi_2(p, s, upstream.kde, damping=0)
    if method == "gedi":
        if p.device.type != "cpu":
            raise ValueError("Upstream GeDI uses CPU gelsd least squares; use CPU.")
        return upstream.gedi(s, p, degree=degree, use_torch=True)
    raise ValueError(method)


class Transform(nn.Module):
    """Independent implementation of Grari's scalar transformation architecture."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16),
                                    nn.ReLU(), nn.Linear(16, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, x):
        return self.layers(x.reshape(-1, 1)).reshape(-1)


class NeuralHGR(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.f, self.g = Transform(), Transform()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, p, s):
        return corr(self.f(p), self.g(s))

    def maximize(self, p, s, steps):
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        for _ in range(steps):
            self.optimizer.zero_grad()
            loss = -self(p.detach(), s.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite adversary loss")
            loss.backward()
            self.optimizer.step()
        # Retain the derivative through p, but freeze adversarial parameters.
        for parameter in self.parameters():
            parameter.requires_grad_(False)
