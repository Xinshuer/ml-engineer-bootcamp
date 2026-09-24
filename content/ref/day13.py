# imports of the drill file (the checks use them)
import sys, os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Day 13 参考答案 —— 注意 AudioLM 和 Day 4 的 GPT 差别有多小。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONVERGED = True


class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


def codec_bitrate(frame_rate, n_q, codebook_size):
    return frame_rate * n_q * math.log2(codebook_size)


def tokens_per_second(frame_rate, n_q):
    return frame_rate * n_q


def code_shape(seconds, frame_rate, n_q):
    return (n_q, int(seconds * frame_rate))


def build_delay(codes, pad_id):
    B, n_q, T = codes.shape
    out = torch.full((B, n_q, T + n_q - 1), pad_id, dtype=codes.dtype, device=codes.device)
    for k in range(n_q):
        out[:, k, k:k + T] = codes[:, k]
    return out


def undo_delay(delayed, T):
    n_q = delayed.size(1)
    return torch.stack([delayed[:, k, k:k + T] for k in range(n_q)], dim=1)


class CodeEmbedding(nn.Module):
    def __init__(self, n_q, vocab, dim):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(vocab, dim) for _ in range(n_q)])

    def forward(self, codes):
        return sum(tab(codes[:, k]) for k, tab in enumerate(self.tables))


class MultiHead(nn.Module):
    def __init__(self, dim, n_q, vocab):
        super().__init__()
        self.n_q, self.vocab = n_q, vocab
        self.lin = nn.Linear(dim, n_q * vocab)

    def forward(self, h):
        B, T, _ = h.shape
        return self.lin(h).view(B, T, self.n_q, self.vocab).permute(0, 2, 1, 3)


def codebook_loss(logits, targets, ignore_index=-100):
    V = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=ignore_index)


class AudioLM(nn.Module):
    def __init__(self, n_q=4, vocab=50, dim=64, n_layer=2, n_head=4, max_len=256):
        super().__init__()
        self.emb = CodeEmbedding(n_q, vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.h = nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = MultiHead(dim, n_q, vocab)

    def forward(self, codes):
        T = codes.size(-1)
        x = self.emb(codes) + self.pos(torch.arange(T, device=codes.device))
        for b in self.h:
            x = b(x)
        return self.head(self.ln_f(x))


def text_cross_attn(audio_h, text_emb, wq, wk, wv, n_head):
    B, Ta, C = audio_h.shape
    Tt = text_emb.size(1)
    hd = C // n_head
    q = (audio_h @ wq).view(B, Ta, n_head, hd).transpose(1, 2)
    k = (text_emb @ wk).view(B, Tt, n_head, hd).transpose(1, 2)
    v = (text_emb @ wv).view(B, Tt, n_head, hd).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v)
    return o.transpose(1, 2).reshape(B, Ta, C)


@torch.no_grad()
def generate_codes(model, prompt, n_new, vocab):
    seq = prompt
    for _ in range(n_new):
        logits = model(seq)                 # (1, n_q, T, V)
        nxt = logits[:, :, -1].argmax(-1)   # (1, n_q)
        seq = torch.cat([seq, nxt.unsqueeze(-1)], dim=-1)
    return seq
