"""Day 14 参考答案。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GRADUATED = True

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
    "训练正常，推理输出全是同一个 token": "忘了model.eval或no_grad",
    "第一个 epoch 正常，第二个 epoch OOM": "循环里存了tensor",
}


def smoke_test(model, *input_shapes, expect_shape=None):
    xs = [torch.randn(s) for s in input_shapes]
    out = model(*xs)
    if expect_shape is not None and tuple(out.shape) != tuple(expect_shape):
        raise AssertionError(f"输出 shape {tuple(out.shape)} != 期望 {tuple(expect_shape)}")
    if not torch.isfinite(out).all():
        raise AssertionError("输出里有 nan / inf")
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
    return z + (e - z).detach(), idx         # 直通：前向是 e，反向绕过量化


def vq_loss(z, z_q_hard, beta=0.25):
    codebook = F.mse_loss(z.detach(), z_q_hard)
    commit = beta * F.mse_loss(z, z_q_hard.detach())
    return codebook + commit


def _toy_data():
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
        # 第一项数值上有意义但对码本没有梯度（见 day14 第 10 题），
        # 真正推动码本的是第二项
        loss = F.mse_loss(zq, data) + vq_loss(data, codebook[idx], beta=0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
