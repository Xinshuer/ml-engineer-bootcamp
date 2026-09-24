"""Day 4 reference module: the nanoGPT pieces (causal mask, attention, CausalSelfAttention, MLP, Block, GPT,
parameter count, greedy generation, and the fixed versions of the two find-the-bug exercises).

Loaded as hidden setup by the day-4 exercises (the names an exercise asks for are removed again before the
learner's code runs). Later days may reuse Block / GPT from here, so keep the names and behaviour stable.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_bias(T, device=None):
    m = torch.tril(torch.ones(T, T, device=device))
    return torch.zeros(1, 1, T, T, device=device).masked_fill(m == 0, float("-inf"))


def attention(q, k, v, causal=True):
    T = q.size(-2)
    att = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        att = att + causal_bias(T, q.device)
    att = att.softmax(dim=-1)
    return att @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(hs)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=2, n_head=4, n_embd=32):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init)
        self.lm_head.weight = self.wte.weight     # weight tying: one shared Parameter object

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)          # (B,T,C) + (T,C) 广播
        for b in self.h:
            x = b(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def gpt_param_count(V, C, L, block_size, tied=True):
    per_block = 12 * C * C + 9 * C          # 12C² 权重 + 9C 的 bias/norm
    n = V * C + block_size * C + L * per_block + 2 * C
    if not tied:
        n += V * C
    return n


@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens, block_size):
    for _ in range(max_new_tokens):
        cond = idx[:, -block_size:]
        logits, _ = model(cond)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def attention_buggy(q, k, v):
    T = q.size(-2)
    att = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))   # 修 1: 漏了缩放
    mask = torch.tril(torch.ones(T, T))                     # 修 2: triu -> tril
    att = att.masked_fill(mask == 0, float("-inf"))
    att = F.softmax(att, dim=-1)                            # 修 3: dim=-2 -> -1
    return att @ v


class BlockBuggy(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # 修 1: post-LN -> pre-LN
        x = x + self.mlp(self.ln2(x))     # 修 2: 漏了残差
        return x
