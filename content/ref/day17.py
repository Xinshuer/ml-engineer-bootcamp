"""Day 17 参考实现：不报错的 bug，以及抓住它们的不变量测试。
Day 17 reference code: bugs that raise no error, and the invariant tests that catch them.

练习的 tests 和卡片的例子会用到这里的小模型；每道题要写的函数（targets）会在运行前被删掉。
The exercises' tests and the cards use the small models below; each item's target function is removed before it runs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- small models / 小模型
def make_classifier():
    """8 维输入 -> 3 类，带 BatchNorm 和 Dropout。 8 inputs -> 3 classes, with BatchNorm and Dropout."""
    return nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 3))


class ModeSpy(nn.Module):
    """不改输入，只记下每次 forward 的 (batch 大小, 是否 train 模式)。
    Passes the input through and records (batch size, training?) for every forward call."""

    def __init__(self):
        super().__init__()
        self.log = []

    def forward(self, x):
        self.log.append((x.shape[0], self.training))
        return x


class TinyAttention(nn.Module):
    """单头 self-attention。mask: "causal"（正确）、"none"（能看到全部）、"peek1"（多看一个未来位置）。
    One-head self-attention. mask: "causal" (correct), "none" (sees everything), "peek1" (sees one future step)."""

    def __init__(self, d=8, mask="causal", p_drop=0.0):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.drop = nn.Dropout(p_drop)
        self.mask = mask

    def forward(self, x):  # (B, T, d)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        T = x.shape[1]
        s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, T, T)
        allowed = torch.ones(T, T, dtype=torch.bool)
        if self.mask == "causal":
            allowed = allowed.tril()
        elif self.mask == "peek1":
            allowed = allowed.tril(diagonal=1)
        s = s.masked_fill(~allowed, float("-inf"))
        return self.drop(s.softmax(-1) @ v)


class CumMean(nn.Module):
    """位置 t 的输出 = 前 t+1 个输入的平均（因果）。 Output at t = mean of inputs 0..t (causal)."""

    def forward(self, x):  # (B, T, d)
        n = torch.arange(1, x.shape[1] + 1, dtype=x.dtype).view(1, -1, 1)
        return x.cumsum(1) / n


class SameConv(nn.Module):
    """kernel 3、padding=1 的 Conv1d：每个位置都能看到下一个位置。 Kernel-3 conv with padding=1: sees the next step."""

    def __init__(self, d=8):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=1)

    def forward(self, x):  # (B, T, d)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class PeekMiddle(nn.Module):
    """几乎是 CumMean，只有位置 2 偷看了位置 3。 Like CumMean, except that position 2 peeks at position 3."""

    def forward(self, x):  # (B, T, d), T >= 4
        out = CumMean()(x)
        out[:, 2] = out[:, 2] + x[:, 3]
        return out


class CopyModel(nn.Module):
    """每个位置都预测「就是当前这个 token」。 Predicts the current token at every position."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, tokens):  # (B, T) -> (B, T, V)
        return 10.0 * F.one_hot(tokens, self.vocab).float()


class NextModel(nn.Module):
    """每个位置都预测「当前 token + 1」。 Predicts (current token + 1) at every position."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, tokens):  # (B, T) -> (B, T, V)
        return 10.0 * F.one_hot((tokens + 1) % self.vocab, self.vocab).float()


# ---------------------------------------------------------------- reference answers / 参考答案
def predict(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return logits.softmax(-1)


def fit(model, train_batches, val_batch, eval_every=2):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    val_losses = []
    for step, (x, y) in enumerate(train_batches):
        model.train()
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = val_batch
                val_losses.append(F.mse_loss(model(xv), yv).item())
    return val_losses


def normalize_split(x, n_train):
    train, test = x[:n_train], x[n_train:]
    mean, std = train.mean(0), train.std(0)
    return (train - mean) / std, (test - mean) / std, mean, std


def lm_loss(model, tokens):
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    V = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))


def masked_lm_loss(logits, targets, mask):
    V = logits.size(-1)
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    mask = mask.reshape(-1).to(loss.dtype)
    return (loss * mask).sum() / mask.sum()


def accum_step(model, opt, micro_batches):
    n = sum(len(x) for x, _ in micro_batches)
    opt.zero_grad()
    total = 0.0
    for x, y in micro_batches:
        loss = F.mse_loss(model(x), y) * (len(x) / n)
        loss.backward()
        total += loss.item()
    opt.step()
    return total


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def train_with_warmup(model, batches, epochs, lr=0.1, warmup=8):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    used = []
    for epoch in range(epochs):
        for x, y in batches:
            loss = F.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            used.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
    return used


def check_causal(model, x):
    model.eval()
    with torch.no_grad():
        base = model(x)
        for t in range(x.shape[1] - 1):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:])
            if not torch.allclose(model(x2)[:, :t + 1], base[:, :t + 1], atol=1e-5):
                return False
    return True


def check_batch_independent(f, x):
    h = x.shape[0] // 2
    with torch.no_grad():
        whole = f(x)
        parts = torch.cat([f(x[:h]), f(x[h:])])
    return torch.allclose(parts, whole, atol=1e-5)
