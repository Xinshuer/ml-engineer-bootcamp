# Day 9 reference module (hidden setup). Needs day 8's module loaded first (`ref: day8, day9`):
# it uses linear_schedule, q_sample and extract from there.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LDM_SHAPES = {
    "image": (4, 3, 512, 512),
    "latent": (4, 4, 64, 64),
    "latent_tokens": (4, 64 * 64, 320),
    "text_emb": (4, 77, 768),
    "cross_attn_out": (4, 64 * 64, 320),
    "unet_out": (4, 4, 64, 64),
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
    k = (ctx @ wk).view(B, Tc, n_head, hd).transpose(1, 2)     # K/V come from the text
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
    both_c = torch.cat([null_cond.expand(B, -1), cond])       # convention: first half null, second half cond
    out = model(both_x, both_t, both_c)
    e_u, e_c = out.chunk(2)
    return cfg_combine(e_u, e_c, scale)


def ddim_timesteps(T, steps):
    # exactly `steps` values, also when T is not a multiple of steps
    return (torch.arange(steps) * (T // steps)).flip(0).long()


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
    both_c = torch.cat([null_cond.expand(cond.size(0), -1), cond])   # fix 1: order [null, cond]
    both_t = torch.cat([t, t])
    out = model(both_x, both_t, both_c)
    e_u, e_c = out.chunk(2)
    return e_u + scale * (e_c - e_u)                                 # fix 2: direction (cond - uncond)


# helper the checks use
class _FakeEps(nn.Module):
    """Fake noise model: output = 0 * x + the mean of the condition vector, so the checks can see which
    condition went into which half of the batch."""
    def forward(self, x, t, c):
        return torch.zeros_like(x) + c.mean(-1).reshape(-1, 1, 1, 1)
