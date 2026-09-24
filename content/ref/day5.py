# imports of the drill file (the checks use them)
import sys, os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Day 05 参考答案。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs(head_dim, max_seq_len, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_seq_len).float(), inv)   # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]        # 相邻两维配对
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2)   # 交错还原


def apply_rope_complex(x, freqs):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    fc = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(xc * fc).flatten(-2).type_as(x)


def swiglu_hidden(n_embd, multiple_of=256):
    h = int(2 * (4 * n_embd) / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    def __init__(self, n_embd, hidden):
        super().__init__()
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, nkv, T, hd = x.shape
    return x[:, :, None].expand(B, nkv, n_rep, T, hd).reshape(B, nkv * n_rep, T, hd)


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    @property
    def length(self):
        return 0 if self.k is None else self.k.size(-2)

    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=-2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=-2)
        return self.k, self.v

    def reset(self):
        self.k = self.v = None


def attn_step(q, k_all, v_all):
    # q 是序列里最新的那个位置，它本来就有权看到全部历史，所以不需要 mask
    att = (q @ k_all.transpose(-2, -1) / math.sqrt(q.size(-1))).softmax(-1)
    return att @ v_all


def _full_attention(x, wq, wk, wv, cos, sin):
    B, T, C = x.shape
    nh, hd = 2, C // 2
    q = (x @ wq).view(B, T, nh, hd).transpose(1, 2)
    k = (x @ wk).view(B, T, nh, hd).transpose(1, 2)
    v = (x @ wv).view(B, T, nh, hd).transpose(1, 2)
    q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
    return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)


def incremental_attention(x, wq, wk, wv, cos, sin):
    B, T, C = x.shape
    nh, hd = 2, C // 2
    cache = KVCache()
    outs = []
    for t in range(T):
        xt = x[:, t:t + 1]
        q = (xt @ wq).view(B, 1, nh, hd).transpose(1, 2)
        k = (xt @ wk).view(B, 1, nh, hd).transpose(1, 2)
        v = (xt @ wv).view(B, 1, nh, hd).transpose(1, 2)
        q = apply_rope(q, cos[t:t + 1], sin[t:t + 1])    # 关键: 用第 t 个位置的角度
        k = apply_rope(k, cos[t:t + 1], sin[t:t + 1])
        K, V = cache.update(k, v)
        o = attn_step(q, K, V)
        outs.append(o.transpose(1, 2).reshape(B, 1, C))
    return torch.cat(outs, dim=1)


def apply_temperature(logits, t):
    if t == 0:
        return logits.masked_fill(logits < logits.max(-1, keepdim=True).values, float("-inf"))
    return logits / t


def top_k_filter(logits, k):
    kth = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    s, idx = logits.sort(dim=-1, descending=True)
    probs = s.softmax(-1)
    cum = probs.cumsum(-1)
    remove = (cum - probs) >= p          # 累计到"这个 token 之前"就已经够了 -> 丢掉
    remove[..., 0] = False               # 至少留一个
    mask = torch.zeros_like(remove).scatter(-1, idx, remove)
    return logits.masked_fill(mask, float("-inf"))


def repetition_penalty(logits, prev_ids, penalty=1.2):
    s = logits.gather(1, prev_ids)
    s = torch.where(s > 0, s / penalty, s * penalty)
    return logits.scatter(1, prev_ids, s)


def apply_rope_buggy(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]              # 修 1: 前后对半 -> 相邻配对
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2)  # 修 2: cat -> 交错 stack+flatten
