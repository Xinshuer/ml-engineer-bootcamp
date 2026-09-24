# imports of the drill file (the checks use them)
import sys, os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Day 02 参考答案。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = {
    "a @ b":                                (8, 16, 32),
    "a + c":                                (8, 16, 64),
    "a.transpose(0, 1)":                    (16, 8, 64),
    "a.view(8, 16, 4, 16)":                 (8, 16, 4, 16),
    "a.view(8, 16, 4, 16).transpose(1, 2)": (8, 4, 16, 16),
    "a.mean(dim=-1)":                       (8, 16),
    "a.mean(dim=-1, keepdim=True)":         (8, 16, 1),
    "d.permute(0, 2, 1, 3)":                (4, 8, 6, 8),
    "d.flatten(2)":                         (4, 6, 64),
    "a * e":                                (8, 16, 64),
    "c + torch.zeros(16, 1)":               (8, 16, 64),
    "a.unsqueeze(1)":                       (8, 1, 16, 64),
    "torch.cat([a, a], dim=-1)":            (8, 16, 128),
}


def split_heads(qkv, n_head):
    B, T, C3 = qkv.shape
    C = C3 // 3
    hs = C // n_head
    q, k, v = qkv.split(C, dim=-1)
    out = [t.view(B, T, n_head, hs).transpose(1, 2) for t in (q, k, v)]
    return out[0], out[1], out[2]


def flatten_heads(x):
    B, nh, T, hs = x.shape
    # transpose 之后内存不连续，view 会报 "view size is not compatible..."
    return x.transpose(1, 2).contiguous().view(B, T, nh * hs)


def my_linear(x, W, b):
    return x @ W.T + b


def my_softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    e = (x - m).exp()
    return e / e.sum(dim=dim, keepdim=True)


def my_cross_entropy(logits, target):
    logp = logits - logits.logsumexp(dim=-1, keepdim=True)
    picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return -picked.mean()


def seq_loss(logits, targets):
    V = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))


def my_layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + eps) * g + b


def my_rmsnorm(x, g, eps=1e-6):
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rms * g


def attn_scores_matmul(q, k):
    return q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))


def attn_scores_einsum(q, k):
    return torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(q.size(-1))


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)          # (B, T, 1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1e-9)


def pick_target_logits(logits, targets):
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class Scaler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer("running_max", torch.zeros(dim))

    def forward(self, x):
        with torch.no_grad():
            self.running_max.copy_(torch.maximum(self.running_max, x.detach().max(0).values))
        return x * self.weight


def manual_grad_mse(x, target):
    return 2 * (x - target) / x.numel()


def collate(samples, pad_id=0):
    B = len(samples)
    Tmax = max(s.numel() for s in samples)
    x = torch.full((B, Tmax), pad_id, dtype=torch.long)
    mask = torch.zeros(B, Tmax, dtype=torch.long)
    for i, s in enumerate(samples):
        x[i, :s.numel()] = s
        mask[i, :s.numel()] = 1
    return x, mask


def train_one_epoch(model, xs, ys, lr=0.1, steps=200):
    opt = torch.optim.SGD(model.parameters(), lr=lr)   # 修 1: 少了括号
    losses = []
    for _ in range(steps):
        logits = model(xs)
        loss = F.cross_entropy(logits, ys)             # 修 2: 分类标签必须是 int64
        opt.zero_grad(set_to_none=True)                # 修 3: 不清零梯度会一直累加
        loss.backward()
        opt.step()
        losses.append(loss.item())                     # 修 4: 存 tensor 会留住整张计算图
    return losses
