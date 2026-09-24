"""Day 10 参考答案 —— 注意 DiTBlock 和 Day 4 的 Block 差别有多小。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from solutions.day08 import timestep_embedding

DICTATION_DONE = True
REUSE_CONFIRMED = True
LOC = {
    "DDPM 的 q_sample + p_sample_step": 9,
    "Flow 的 flow_interpolate + 欧拉一步": 3,
}


class SelfAttention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, T, C))


def patchify(x, p):
    B, C, H, W = x.shape
    nh, nw = H // p, W // p
    x = x.reshape(B, C, nh, p, nw, p)
    x = x.permute(0, 2, 4, 1, 3, 5)            # (B, nh, nw, C, p, p)
    return x.reshape(B, nh * nw, C * p * p)


def unpatchify(tok, p, H, W, C):
    B = tok.size(0)
    nh, nw = H // p, W // p
    x = tok.reshape(B, nh, nw, C, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C, H, W)


def _pe_1d(dim, pos):
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    a = pos.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


def pos_embed_2d(dim, h, w):
    half = dim // 2
    row = _pe_1d(half, torch.arange(h))         # (h, half)
    col = _pe_1d(half, torch.arange(w))         # (w, half)
    r = row[:, None, :].expand(h, w, half)
    c = col[None, :, :].expand(h, w, half)
    return torch.cat([r, c], dim=-1).reshape(h * w, dim)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.lin = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.lin.weight)      # Zero 的来历
        nn.init.zeros_(self.lin.bias)

    def forward(self, c):
        return self.lin(self.act(c)).chunk(6, dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, n_head)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.modulation = AdaLNModulation(dim)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.modulation(c)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


def flow_interpolate(x0_noise, x1_data, t):
    tt = t.reshape(-1, *([1] * (x0_noise.dim() - 1)))
    return (1 - tt) * x0_noise + tt * x1_data


def flow_target(x0_noise, x1_data):
    return x1_data - x0_noise


def flow_loss(model, x1_data):
    B = x1_data.size(0)
    t = torch.rand(B, device=x1_data.device)
    noise = torch.randn_like(x1_data)
    xt = flow_interpolate(noise, x1_data, t)
    return F.mse_loss(model(xt, t), flow_target(noise, x1_data))


@torch.no_grad()
def flow_sample(model, shape, steps=20, device="cpu"):
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        x = x + model(x, t) * dt
    return x


class MiniDiT(nn.Module):
    def __init__(self, in_ch=4, img=8, patch=2, dim=64, n_layer=2, n_head=4):
        super().__init__()
        self.in_ch, self.img, self.patch, self.dim = in_ch, img, patch, dim
        pdim = patch * patch * in_ch
        self.embed = nn.Linear(pdim, dim)
        n = (img // patch) ** 2
        self.register_buffer("pos", pos_embed_2d(dim, img // patch, img // patch))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, n_head) for _ in range(n_layer)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod_out = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.mod_out.weight)
        nn.init.zeros_(self.mod_out.bias)
        self.head = nn.Linear(dim, pdim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, t):
        h = self.embed(patchify(x, self.patch)) + self.pos
        c = self.t_mlp(timestep_embedding(t, self.dim))
        for b in self.blocks:
            h = b(h, c)
        shift, scale = self.mod_out(F.silu(c)).chunk(2, dim=-1)
        h = self.head(modulate(self.norm_out(h), shift, scale))
        return unpatchify(h, self.patch, self.img, self.img, self.in_ch)


class DiTBlockBuggy(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)   # 修 4
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)   # 修 4
        self.attn = SelfAttention(dim, n_head)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[1].weight)                  # 修 1
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.modulation(c).chunk(6, dim=-1)
        # 修 2: 补上 gate；修 3: scale 要写成 (1 + scale)
        x = x + g1.unsqueeze(1) * self.attn(
            self.norm1(x) * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1))
        x = x + g2.unsqueeze(1) * self.mlp(
            self.norm2(x) * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1))
        return x
