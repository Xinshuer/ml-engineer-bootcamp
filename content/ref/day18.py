"""Day 18 reference code (hidden setup): train/serve skew.

Shared helpers for the items and cards that list `ref: day18`. Names an item lists in `targets:` (here: left_pad)
are removed again after this module runs.
"""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------- a small classifier with BatchNorm + Dropout
def make_net(seed=0):
    """Linear -> BatchNorm -> ReLU -> Dropout -> Linear. BN running stats are warmed up, so eval() differs
    from train(). Returned in train mode (the state a freshly built / freshly trained model is in)."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 3))
    with torch.no_grad():
        for _ in range(20):
            net(torch.randn(32, 4) * 2 + 1)
    net.train()
    return net


# ---------------------------------------------------------------- a tiny decoder-only LM for batched generation
class TinyLM(nn.Module):
    """Token + learned position embeddings, 2 causal self-attention blocks (1 head), a head over the vocab.

    forward(ids, attention_mask=None, position_ids=None) -> logits (B, T, V)
      attention_mask: (B, T), 1 = real token, 0 = padding (keys that are padding are never attended to)
      position_ids:   (B, T), the position of each token inside its own sequence
    Masked scores are filled with -1e9 (not -inf), so a row that is all padding gives no NaN.
    """

    def __init__(self, vocab=16, d=32, max_len=32, n_layers=2):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d), "qkv": nn.Linear(d, 3 * d), "proj": nn.Linear(d, d),
                "ln2": nn.LayerNorm(d), "mlp": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
            }))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids, attention_mask=None, position_ids=None):
        B, T = ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(ids)
        if position_ids is None:
            position_ids = torch.arange(T, device=ids.device).expand(B, T)
        x = self.tok(ids) + self.pos(position_ids)
        causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
        allowed = causal[None] & attention_mask.bool()[:, None, :]          # (B, T, T)
        for blk in self.blocks:
            q, k, v = blk["qkv"](blk["ln1"](x)).chunk(3, dim=-1)
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.d)
            scores = scores.masked_fill(~allowed, -1e9)
            x = x + blk["proj"](scores.softmax(-1) @ v)
            x = x + blk["mlp"](blk["ln2"](x))
        return self.head(self.ln_f(x))


def tiny_lm(seed=0):
    """A TinyLM with fixed random weights, in eval mode. Token 0 is the pad token; it is never generated."""
    torch.manual_seed(seed)
    model = TinyLM().eval()
    with torch.no_grad():
        model.head.weight[0] = 0.0
        model.head.bias[0] = -100.0
    return model


def generate_one(model, prompt, n_new):
    """Greedy generation for ONE prompt (no padding at all): the reference that batched generation must match."""
    ids = torch.tensor([prompt])
    new = []
    with torch.inference_mode():
        for _ in range(n_new):
            nxt = model(ids)[:, -1].argmax(-1)
            new.append(int(nxt))
            ids = torch.cat([ids, nxt[:, None]], dim=1)
    return new


def left_pad(seqs, pad_id=0):
    """[[5, 6, 7], [8]] -> ids [[5, 6, 7], [0, 0, 8]], mask [[1, 1, 1], [0, 0, 1]], pos [[0, 1, 2], [0, 0, 0]]"""
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, T - len(s):] = 1
    pos = (mask.cumsum(dim=-1) - 1).clamp(min=0)
    return ids, mask, pos
