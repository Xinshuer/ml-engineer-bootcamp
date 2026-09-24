"""Day 09 参考答案。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from solutions.day08 import extract, linear_schedule, q_sample

LDM_SHAPES = {
    "图像 x": (4, 3, 512, 512),
    "VAE 编码后的 latent": (4, 4, 64, 64),
    "latent 转成 token 序列": (4, 64 * 64, 320),
    "CLIP 文本嵌入": (4, 77, 768),
    "cross-attn 输出": (4, 64 * 64, 320),
    "UNet 输出（预测的噪声）": (4, 4, 64, 64),
}


def to_tokens(x):
    return x.flatten(2).transpose(1, 2)          # (B,C,H,W) -> (B,HW,C)


def to_spatial(tok, H, W):
    B, N, C = tok.shape
    return tok.transpose(1, 2).reshape(B, C, H, W)


def cross_attention(x, ctx, wq, wk, wv, n_head):
    B, Tx, C = x.shape
    Tc = ctx.size(1)
    hd = C // n_head
    q = (x @ wq).view(B, Tx, n_head, hd).transpose(1, 2)
    k = (ctx @ wk).view(B, Tc, n_head, hd).transpose(1, 2)     # KV 来自文本
    v = (ctx @ wv).view(B, Tc, n_head, hd).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v)
    return o.transpose(1, 2).reshape(B, Tx, C)


def combine_cond(t_emb, class_emb):
    return t_emb + class_emb


def drop_cond(cond_emb, null_emb, p, generator=None):
    B = cond_emb.size(0)
    keep = (torch.rand(B, 1, generator=generator) >= p).to(cond_emb.dtype)
    return keep * cond_emb + (1 - keep) * null_emb.expand_as(cond_emb)


def cfg_combine(eps_uncond, eps_cond, scale):
    return eps_uncond + scale * (eps_cond - eps_uncond)


def cfg_forward(model, x, t, cond, null_cond, scale):
    B = x.size(0)
    both_x = torch.cat([x, x])
    both_t = torch.cat([t, t])
    both_c = torch.cat([null_cond.expand(B, -1), cond])       # 约定: 前 null 后 cond
    out = model(both_x, both_t, both_c)
    e_u, e_c = out.chunk(2)
    return cfg_combine(e_u, e_c, scale)


def ddim_timesteps(T, steps):
    return torch.arange(0, T, T // steps).flip(0).long()


def ddim_step(x_t, t, t_prev, eps_pred, sched):
    ab = extract(sched["alphas_cumprod"], t, x_t.shape)
    x0 = (x_t - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
    prev_ok = (t_prev >= 0)
    tp = t_prev.clamp(min=0)
    ab_prev = extract(sched["alphas_cumprod"], tp, x_t.shape)
    ones = torch.ones_like(ab_prev)
    ab_prev = torch.where(prev_ok.reshape(-1, *([1] * (x_t.dim() - 1))), ab_prev, ones)
    return ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps_pred


@torch.no_grad()
def ddim_sample(model, shape, sched, steps=10, device="cpu"):
    ts = ddim_timesteps(len(sched["betas"]), steps).to(device)
    x = torch.randn(shape, device=device)
    for i, tv in enumerate(ts):
        t = torch.full((shape[0],), int(tv), dtype=torch.long, device=device)
        tp_val = int(ts[i + 1]) if i + 1 < len(ts) else -1
        t_prev = torch.full((shape[0],), tp_val, dtype=torch.long, device=device)
        x = ddim_step(x, t, t_prev, model(x, t), sched)
    return x


def cfg_sample_buggy(model, x, t, cond, null_cond, scale):
    both_x = torch.cat([x, x])
    both_c = torch.cat([null_cond.expand(cond.size(0), -1), cond])   # 修 1: 顺序
    both_t = torch.cat([t, t])
    out = model(both_x, both_t, both_c)
    e_u, e_c = out.chunk(2)
    return e_u + scale * (e_c - e_u)                                 # 修 2+3
