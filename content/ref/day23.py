"""Day 23 参考实现：生成式模型的推理服务。
Day 23 reference code: serving generative models.

练习的 tests 和卡片的例子会用到这里的小模型和工具；每道题要写的函数（targets）会在运行前被删掉。
The exercises' tests and the cards use the small models and helpers below; each item's target is removed before it runs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- a tiny random causal LM with a KV cache
class _Block(nn.Module):
    """Pre-norm Transformer block with GQA attention: n_heads query heads share n_kv_heads k/v heads."""

    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.hd = n_heads, n_kv_heads, d // n_heads
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, n_heads * self.hd, bias=False)
        self.kv = nn.Linear(d, 2 * n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(n_heads * self.hd, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, past, start):
        B, T, _ = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, T, self.n_heads, self.hd).transpose(1, 2)            # (B, nh, T, hd)
        k, v = self.kv(h).view(B, T, 2, self.n_kv_heads, self.hd).permute(2, 0, 3, 1, 4)
        k, v = k.contiguous(), v.contiguous()                                      # (B, nkv, T, hd)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        rep = self.n_heads // self.n_kv_heads
        kk, vv = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        allowed = torch.arange(k.shape[2]) <= (start + torch.arange(T))[:, None]    # (T, S): only the past
        a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=allowed)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        x = x + self.mlp(self.ln2(x))
        return x, (k, v)


class TinyLM(nn.Module):
    """很小的随机因果语言模型，带 KV cache。 A tiny random causal LM with a KV cache.

        logits, cache = model(ids, cache)

    ids: (B, T) long. cache: None 表示 prefill / None means prefill; otherwise what the previous call returned.
    logits: (B, T, vocab). cache: 每层一个 (k, v) / one (k, v) per layer, each (B, n_kv_heads, tokens so far, head_dim).
    model.calls: 每次调用处理了几个 token / how many tokens each call processed.
    """

    def __init__(self, vocab=32, d=32, n_layers=2, n_heads=4, n_kv_heads=2, max_len=128):
        super().__init__()
        self.vocab, self.max_len = vocab, max_len
        self.n_layers, self.n_kv_heads, self.head_dim = n_layers, n_kv_heads, d // n_heads
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(_Block(d, n_heads, n_kv_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():                          # small positions, equal-norm output rows: the next token depends on the context
            self.pos.weight.mul_(0.2)
            w = torch.randn(vocab, d)
            self.head.weight.copy_(3.0 * w / w.norm(dim=1, keepdim=True))
        self.calls = []

    def forward(self, ids, cache=None):
        B, T = ids.shape
        start = 0 if cache is None else cache[0][0].shape[2]
        if start + T > self.max_len:
            raise ValueError(f"sequence of {start + T} tokens is longer than max_len={self.max_len}")
        self.calls.append(T)
        x = self.tok(ids) + self.pos(torch.arange(start, start + T))
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, None if cache is None else cache[i], start)
            new_cache.append(kv)
        return self.head(self.ln_f(x)), new_cache


def make_tiny_lm(seed=0, **kw):
    """同一个 seed 永远得到同一个模型；不影响全局随机数。 Same seed -> same model; the global RNG is not touched."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyLM(**kw)
    return model.eval()


class ScriptedLM(nn.Module):
    """不管输入是什么，第 n 次调用（从 0 数）都预测 script[n]，用完后一直重复最后一个。接口和 TinyLM 一样。
    Whatever the input, call number n (from 0) predicts script[n], then keeps repeating the last one. Same API as TinyLM."""

    def __init__(self, script, vocab=32):
        super().__init__()
        self.script, self.vocab, self.calls = list(script), vocab, []

    def forward(self, ids, cache=None):
        step = len(self.calls)
        self.calls.append(ids.shape[1])
        logits = torch.zeros(ids.shape[0], ids.shape[1], self.vocab)
        logits[:, -1, self.script[min(step, len(self.script) - 1)]] = 5.0
        return logits, {"calls": step + 1}


@torch.inference_mode()
def greedy_no_cache(model, prompt_ids, n):
    """参考答案：每一步都把整个序列重新算一遍（慢，但一定对）。 Reference: recompute the whole sequence every step."""
    ids, out = prompt_ids, []
    for _ in range(n):
        logits, _ = model(ids)
        tok = logits[0, -1].argmax().item()
        out.append(tok)
        ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
    return out


def cache_bytes(cache):
    """一个 TinyLM cache 实际占的字节数。 Bytes actually held by a TinyLM cache."""
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)


# ---------------------------------------------------------------- KV cache arithmetic
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, dtype):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * dtype.itemsize   # 2 = K and V


# ---------------------------------------------------------------- streaming with stop conditions
@torch.inference_mode()
def stream_tokens(model, prompt_ids, max_new_tokens, eos_id):
    ids, cache = prompt_ids, None
    for _ in range(max_new_tokens):
        logits, cache = model(ids, cache)            # first call: prefill the prompt; then: decode one token
        tok = logits[0, -1].argmax().item()
        if tok == eos_id:
            return
        yield tok
        ids = torch.tensor([[tok]], device=prompt_ids.device)


class CountingIter:
    """包一层迭代器，记下被读走了几个元素。 Wraps an iterable and counts how many items were taken."""

    def __init__(self, items):
        self._it = iter(items)
        self.taken = 0

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self._it)
        self.taken += 1
        return x


def text_until_stop(pieces, stops):
    text = ""
    for piece in pieces:
        text += piece
        hits = [text.find(s) for s in stops if s in text]
        if hits:
            return text[:min(hits)]
    return text


def held_back(text, stops):
    """text 末尾最长的、可能是某个 stop 开头的那一段的长度。
    Length of the longest end of `text` that could be the start of a stop string."""
    keep = 0
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:k]):
                keep = max(keep, k)
                break
    return keep


def stream_until_stop(pieces, stops):
    buf = ""
    for piece in pieces:
        buf += piece
        hits = [buf.find(s) for s in stops if s in buf]
        if hits:
            if buf[:min(hits)]:
                yield buf[:min(hits)]
            return
        keep = held_back(buf, stops)
        if len(buf) > keep:
            yield buf[:len(buf) - keep]
            buf = buf[len(buf) - keep:]
    if buf:
        yield buf


# ---------------------------------------------------------------- sampling parameters
def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate_sampling(params, prompt_len, context_len):
    errors = []
    for key in params:
        if key not in ("temperature", "top_p", "top_k", "max_new_tokens"):
            errors.append(f"{key}: unknown parameter")
    if "temperature" in params:
        t = params["temperature"]
        if not (_is_number(t) and math.isfinite(t) and 0 <= t <= 2):
            errors.append(f"temperature: must be a number from 0 to 2, got {t!r}")
    if "top_p" in params:
        p = params["top_p"]
        if not (_is_number(p) and math.isfinite(p) and 0 < p <= 1):
            errors.append(f"top_p: must be a number in (0, 1], got {p!r}")
    if "top_k" in params:
        k = params["top_k"]
        if not (_is_int(k) and k >= 0):
            errors.append(f"top_k: must be an integer >= 0, got {k!r}")
    if "max_new_tokens" in params:
        n = params["max_new_tokens"]
        if not (_is_int(n) and n >= 1):
            errors.append(f"max_new_tokens: must be an integer >= 1, got {n!r}")
        elif prompt_len + n > context_len:
            errors.append(f"max_new_tokens: prompt ({prompt_len}) + {n} > context length {context_len}")
    return errors


# ---------------------------------------------------------------- batching schedulers (time = decode steps)
def simulate_static(lengths, n_slots):
    """静态 batching：每 n_slots 个请求一批，整批做完才一起返回、才开始下一批。
    Static batching: n_slots requests per batch; the whole batch returns together, then the next batch starts."""
    finish, start = [0] * len(lengths), 0
    for g in range(0, len(lengths), n_slots):
        group = range(g, min(g + n_slots, len(lengths)))
        end = start + max(lengths[i] for i in group)
        for i in group:
            finish[i] = end
        start = end
    return finish


def continuous_timeline(lengths, n_slots):
    """每一步结束时 slot 里是谁、队列里是谁、谁刚做完。 Per step: who is in the slots, who waits, who just finished."""
    queue, slots, done, rows, step = list(range(len(lengths))), [], [0] * len(lengths), [], 0
    while queue or slots:
        step += 1
        while queue and len(slots) < n_slots:
            slots.append(queue.pop(0))
        running = list(slots)
        finished = []
        for i in running:
            done[i] += 1
            if done[i] == lengths[i]:
                finished.append(i)
        slots = [i for i in slots if i not in finished]
        rows.append((step, running, list(queue), finished))
    return rows


def simulate_continuous(lengths, n_slots):
    finish = [0] * len(lengths)
    for step, _, _, finished in continuous_timeline(lengths, n_slots):
        for i in finished:
            finish[i] = step
    return finish


# ---------------------------------------------------------------- diffusion: steps vs latency
def pick_steps(budget_ms, step_ms, overhead_ms, cfg, max_steps=50):
    per_step = step_ms * (2 if cfg else 1)
    steps = math.floor((budget_ms - overhead_ms) / per_step)
    return int(max(0, min(max_steps, steps)))


class TinyDenoiser(nn.Module):
    """扩散模型里「每一步调用一次」的那个网络的迷你版。 A miniature of the network a diffusion sampler calls once per step."""

    def __init__(self, ch=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch, hidden, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(hidden, ch, 3, padding=1))

    def forward(self, x):
        return self.net(x)
