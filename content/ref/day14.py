"""Day 14 reference module (hidden setup for the exercises; the names an item asks for are removed again).

从论文到代码：五问法、smoke test、过拟合一个 batch、梯度裁剪、VQ（直通估计器 + 两个 stop-gradient 项）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FIVE_QUESTIONS = {
    "输入 shape": (8, 3, 64, 64),
    "输出 shape": (8, 3, 64, 64),
    "潜变量 shape": (8, 16, 16),
    "loss 有几项": 3,
    "条件注入方式": "none",
    "采样方式": "autoregressive",
}

DIAGNOSIS = {
    "loss 稳定卡在 ln(vocab_size)": "模型没学到东西",
    "loss 第 3 步就变成 nan": "lr过大或fp16溢出",
    "训练 loss 掉到 0.001，生成全是乱码": "标签泄漏",
    "训练正常，贪心解码时同一个输入每次输出都不一样": "忘了model.eval",
    "第一个 epoch 正常，第二个 epoch OOM": "循环里存了带计算图的tensor",
}


def smoke_test(model, *input_shapes, expect_shape=None):
    """Feed random tensors once; check the output shape and that every value is finite."""
    xs = [torch.randn(s) for s in input_shapes]
    out = model(*xs)
    if expect_shape is not None and tuple(out.shape) != tuple(expect_shape):
        raise AssertionError(f"output shape {tuple(out.shape)} != expected {tuple(expect_shape)}")
    if not torch.isfinite(out).all():
        raise AssertionError("the output contains nan / inf")
    return True


def overfit_one_batch(model, x, y, steps=300, lr=1e-2):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def clip_and_report(model, max_norm):
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def vq_quantize(z, codebook):
    d = torch.cdist(z, codebook)             # (B, K)
    idx = d.argmin(-1)
    e = codebook[idx]
    return z + (e - z).detach(), idx         # straight-through: forward is e, backward skips the quantization


def vq_loss(z, z_q_hard, beta=0.25):
    codebook = F.mse_loss(z.detach(), z_q_hard)          # ||sg[z] - e||^2: only the codebook gets gradients
    commit = beta * F.mse_loss(z, z_q_hard.detach())     # beta * ||z - sg[e]||^2: only the encoder gets gradients
    return codebook + commit


def _toy_data():
    """64 one-dimensional points around 4 centres [-3, -1, 1, 3], 16 each, noise std 0.1. Shape (64, 1)."""
    torch.manual_seed(42)
    c = torch.tensor([[-3.], [-1.], [1.], [3.]])
    return c.repeat_interleave(16, 0) + torch.randn(64, 1) * 0.1


def train_vq_toy(steps=300, lr=1e-1):
    data = _toy_data()
    codebook = nn.Parameter(torch.linspace(-0.3, 0.3, 4).reshape(4, 1))
    opt = torch.optim.AdamW([codebook], lr=lr)
    losses = []
    for _ in range(steps):
        zq, idx = vq_quantize(data, codebook)
        # the first term has a value but no gradient to the codebook; the second term moves the codebook
        loss = F.mse_loss(zq, data) + vq_loss(data, codebook[idx], beta=0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
