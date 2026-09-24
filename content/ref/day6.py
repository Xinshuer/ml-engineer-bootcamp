# imports of the drill file (the checks use them)
import sys, os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Day 06 参考答案。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features).normal_(0, 0.01))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # 必须是 0
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def merge_lora(m):
    out_f, in_f = m.base.out_features, m.base.in_features
    lin = nn.Linear(in_f, out_f, bias=m.base.bias is not None)
    with torch.no_grad():
        lin.weight.copy_(m.base.weight + m.scaling * (m.B @ m.A))
        if m.base.bias is not None:
            lin.bias.copy_(m.base.bias)
    return lin


def lora_ratio(in_f, out_f, r):
    return (r * in_f + out_f * r) / (in_f * out_f)


def build_sft_labels(input_ids, prompt_len, ignore_index=-100):
    lab = input_ids.clone().long()
    lab[:prompt_len] = ignore_index
    return lab


def apply_chat_template(messages, add_generation_prompt=True):
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:
        s += "<|im_start|>assistant\n"
    return s


def grad_accum_step(model, xs, ys, n_micro):
    model.zero_grad(set_to_none=True)
    n = xs.size(0) // n_micro
    for i in range(n_micro):
        xb, yb = xs[i * n:(i + 1) * n], ys[i * n:(i + 1) * n]
        # 每份内部已经对自己的 batch 求了平均，再除以份数才等于整体平均
        (F.cross_entropy(model(xb), yb) / n_micro).backward()
    return {k: p.grad.clone() for k, p in model.named_parameters()}


def get_lr(it, warmup_iters, max_iters, max_lr, min_lr):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    progress = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def optimizer_memory_bytes(n_params, dtype_bytes=4, optimizer="adamw"):
    mult = {"adamw": 4, "sgd": 2, "sgd_momentum": 3}[optimizer]
    return n_params * dtype_bytes * mult


class LoRABuggy(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():      # 修 1: base 要冻结
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # 修 2: B 初始化为 0
        self.scaling = alpha / r                                   # 修 3: 除不是乘

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling
