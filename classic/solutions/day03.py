"""Day 03 参考答案。"""
import math

import numpy as np
import torch

BATCH_SHAPES = {
    "x  (输入 token id)":         (8, 256),
    "y  (标签)":                  (8, 256),
    "tok_emb = wte(x)":           (8, 256, 768),
    "pos_emb = wpe(pos)":         (256, 768),
    "hidden (每个 block 的输出)":  (8, 256, 768),
    "logits = lm_head(hidden)":   (8, 256, 50257),
    "logits.view(-1, V)":         (8 * 256, 50257),
    "y.view(-1)":                 (8 * 256,),
    "loss":                       (),
}


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.itos = {i: c for i, c in enumerate(chars)}
        self.stoi = {c: i for i, c in self.itos.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def pair_counts(tokens):
    c = {}
    for a, b in zip(tokens, tokens[1:]):
        c[(a, b)] = c.get((a, b), 0) + 1
    return c


def merge_pair(tokens, pair):
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            out.append(tokens[i] + tokens[i + 1])
            i += 2                      # 跳过两个，重叠的 pair 不会被重复合并
        else:
            out.append(tokens[i])
            i += 1
    return out


def compression_ratio(text, encode_fn):
    return len(text) / len(encode_fn(text))


def get_batch(data, batch_size, block_size, generator=None):
    # 上界是 len-block_size，这样最大起点 i = len-block_size-1，y 的末位刚好是 len-1
    hi = len(data) - block_size
    ix = torch.randint(hi, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).long()
    return x, y


def save_bin(path, ids):
    np.asarray(ids, dtype=np.uint16).tofile(path)


def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def bigram_counts(ids, vocab_size):
    flat = ids[:-1].long() * vocab_size + ids[1:].long()
    return torch.bincount(flat, minlength=vocab_size ** 2).reshape(vocab_size, vocab_size).float()


def uniform_loss(vocab_size):
    return math.log(vocab_size)


def is_broken(loss, vocab_size, tol=0.05):
    base = uniform_loss(vocab_size)
    return abs(loss - base) / base < tol


def pad_batch(seqs, pad_id=0):
    B, Tmax = len(seqs), max(len(s) for s in seqs)
    ids = torch.full((B, Tmax), pad_id, dtype=torch.long)
    mask = torch.zeros(B, Tmax, dtype=torch.long)
    pos = torch.zeros(B, Tmax, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = len(s)
        ids[i, :L] = torch.tensor(s, dtype=torch.long)
        mask[i, :L] = 1
        pos[i, :L] = torch.arange(L)
    return ids, mask, pos


def mask_to_bias(attn_mask):
    b = torch.zeros(attn_mask.size(0), 1, 1, attn_mask.size(1))
    # 注意不能写 (1-mask)*(-inf)，0 * inf = nan
    return b.masked_fill(attn_mask[:, None, None, :] == 0, float("-inf"))


def get_batch_buggy(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))   # 修 1: 起点越界
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])  # 修 2: y 忘了右移
    return x.long(), y.long()                                   # 修 3: 不能转 float
