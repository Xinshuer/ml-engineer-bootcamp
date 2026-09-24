"""Day 15 reference module (hidden setup): toy data, tiny models, small test layers, and the reference answers.

Every exercise with `ref: day15` gets these names; the names an exercise asks the learner to write are removed
again before the learner's code runs. Names starting with `_` are helpers for the checks.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- toy data and tiny models

def toy_data(n=32, d=8, c=4, seed=0):
    """n samples, d features, c classes. y = argmax(x @ w), so a tiny MLP can learn (and memorise) it."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, c, generator=g)
    return x, (x @ w).argmax(-1)


def tiny_mlp(d=8, h=32, c=4, seed=0):
    """Linear -> ReLU -> Linear. Seeded, so two calls give the same starting weights."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, c))


class TinyNet(nn.Module):
    """A 'pretrained' model: backbone (features) + head (classifier)."""

    def __init__(self, d=8, h=32, c=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.backbone = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.head = nn.Linear(h, c)

    def forward(self, x):
        return self.head(self.backbone(x))


class Log(nn.Module):
    """log(x): negative inputs give nan, 0 gives -inf."""

    def forward(self, x):
        return torch.log(x)


class Exp(nn.Module):
    """exp(x): inputs above about 88.7 give inf in float32."""

    def forward(self, x):
        return torch.exp(x)


class Scale(nn.Module):
    """Multiplies by a constant (like a badly chosen temperature or init scale)."""

    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x * self.s


# ---------------------------------------------------------------- reference answers

def curve_verdict(losses):
    if not all(math.isfinite(v) for v in losses):
        return "nan"
    if losses[-1] > 2 * losses[0]:
        return "exploding"
    if abs(losses[-1] - losses[0]) < 0.01 * losses[0]:
        return "flat"
    return "ok"


def my_cross_entropy(logits, y):
    logp = F.log_softmax(logits, dim=-1)
    return -logp[torch.arange(len(y)), y].mean()


def rms_norm(x, weight, eps=1e-6):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return out.to(x.dtype) * weight


def find_nan_layer(model, x):
    for name, layer in model.named_children():
        x = layer(x)
        if not torch.isfinite(x).all():
            return name
    return None


def pairwise_dist(a, b):
    diff = a[:, None, :] - b[None, :, :]
    return (diff.pow(2).sum(dim=-1) + 1e-12).sqrt()


def grad_norms(model):
    return {name: (None if p.grad is None else p.grad.norm().item())
            for name, p in model.named_parameters()}


def train_classifier(model, x, y, steps=100, lr=0.5):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        loss = F.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step {step}: loss = {loss.item()}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def make_batches(x, y, batch_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    x, y = x[perm], y[perm]
    return [(x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(x), batch_size)]


def finetune_new_head(model, x, y, num_classes, steps=100, lr=0.5):
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.head = nn.Linear(model.head.in_features, num_classes)
    opt = torch.optim.SGD(model.head.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------- helpers for the checks

def _sgd_ref(model, x, y, steps, lr):
    """Plain full-batch SGD + cross_entropy, used to check a learner's training loop step by step."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def _rms_ref(x, weight, eps=1e-6):
    """RMSNorm computed in float64 (no overflow), returned as float32."""
    xd = x.double()
    out = xd * torch.rsqrt(xd.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.double()
    return out.float()
