# imports of the drill file (the checks use them)
import sys, os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Day 08 参考答案 —— UNet 是全课最长的一段，先抄一遍跑通，第二遍默写。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    return {"betas": betas, "alphas": alphas, "alphas_cumprod": alphas.cumprod(0)}


def cosine_schedule(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0])[1:]
    prev = torch.cat([torch.ones(1), ab[:-1]])
    betas = (1 - ab / prev).clamp(0, 0.999)
    return {"betas": betas, "alphas": 1 - betas, "alphas_cumprod": ab}


def extract(arr, t, x_shape):
    out = arr.to(t.device).gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))


def q_sample(x0, t, noise, sched):
    ab = extract(sched["alphas_cumprod"], t, x0.shape)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    # device 必须跟着 t 走，否则模型搬到 GPU 之后这里会报 "two devices" —— 
    # 这类 bug 在 CPU 上永远测不出来
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class FiLM(nn.Module):
    def __init__(self, t_dim, ch):
        super().__init__()
        self.proj = nn.Linear(t_dim, 2 * ch)

    def forward(self, h, emb):
        scale, shift = self.proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        return h * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.film = FiLM(t_dim, out_ch)
        self.n2 = nn.GroupNorm(8, out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = self.film(h, emb)
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


def unet_skip_concat(up_feat, skip_feat):
    dh = skip_feat.size(-2) - up_feat.size(-2)
    dw = skip_feat.size(-1) - up_feat.size(-1)
    if dh or dw:
        up_feat = F.pad(up_feat, (0, dw, 0, dh))
    return torch.cat([up_feat, skip_feat], dim=1)


class MiniUNet(nn.Module):
    def __init__(self, in_ch=1, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(64, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.stem = nn.Conv2d(in_ch, 32, 3, 1, 1)
        self.rb1 = ResBlock(32, 32, t_dim)
        self.ds1 = nn.Conv2d(32, 32, 4, 2, 1)
        self.rb2 = ResBlock(32, 64, t_dim)
        self.ds2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.mid = ResBlock(64, 64, t_dim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.rb3 = ResBlock(64 + 64, 32, t_dim)
        self.rb4 = ResBlock(32 + 32, 32, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, in_ch, 3, 1, 1))

    def forward(self, x, t):
        emb = self.t_mlp(timestep_embedding(t, 64))
        h = self.stem(x)
        s1 = self.rb1(h, emb)
        h = self.ds1(s1)
        s2 = self.rb2(h, emb)
        h = self.ds2(s2)
        h = self.mid(h, emb)
        h = self.rb3(unet_skip_concat(self.up(h), s2), emb)
        h = self.rb4(unet_skip_concat(self.up(h), s1), emb)
        return self.out(h)


def ddpm_loss(model, x0, sched, T):
    t = torch.randint(0, T, (x0.size(0),), device=x0.device)
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise, sched)
    return F.mse_loss(model(xt, t), noise)      # 目标是噪声，不是 x0


def p_sample_step(x_t, t, eps_pred, sched, noise=None):
    beta = extract(sched["betas"], t, x_t.shape)
    alpha = extract(sched["alphas"], t, x_t.shape)
    ab = extract(sched["alphas_cumprod"], t, x_t.shape)
    mean = (x_t - beta / (1 - ab).sqrt() * eps_pred) / alpha.sqrt()
    if noise is None:
        noise = torch.randn_like(x_t)
    keep = (t > 0).float().reshape(-1, *([1] * (x_t.dim() - 1)))
    return mean + keep * beta.sqrt() * noise    # t=0 时不加噪声


def snr(sched, t):
    ab = sched["alphas_cumprod"].gather(0, t)
    return ab / (1 - ab)


def q_sample_buggy(x0, t, noise, sched):
    ab = extract(sched["alphas_cumprod"], t, x0.shape)   # 修 1: alphas -> alphas_cumprod
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise      # 修 2: 两个系数都要开根号
