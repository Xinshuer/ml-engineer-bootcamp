"""通用训练循环 —— Day 2 之后 12 天全部复用它。

设计上只有一条原则：**Trainer 不许知道你在训什么**。
它只管步数、学习率、混合精度、梯度累积、裁剪、EMA、日志、checkpoint；
「什么是 loss」永远由你传进来的 loss_fn 决定。所以同一个 Trainer 能跑
分类 / GPT / VAE / DDPM / Flow / TTS / AudioLM，一行都不用改。

    from common import TrainConfig, Trainer

    def loss_fn(model, batch):
        x, y = batch
        return F.cross_entropy(model(x), y)

    Trainer(model, loss_fn, train_data, val_data, TrainConfig(max_steps=500)).fit()

train_data 可以是两种东西之一：
  1. DataLoader 或任何可迭代对象  —— Trainer 会自动无限循环
  2. 一个无参函数 () -> batch     —— LLM / 扩散那种随机采样的写法

loss_fn 返回 loss，或者 (loss, {"名字": 数值}) 让额外指标也进日志。
"""
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import TrainConfig
from common.utils import (EMA, AvgMeter, autocast_ctx, ckpt_path, cuda_mem, describe,
                          load_ckpt, pick_device, save_ckpt, set_seed, to_device)


# ---------------------------------------------------------------- 学习率

def get_lr(step, cfg):
    """线性 warmup + 余弦衰减。三段式，和 nanoGPT / Llama 的写法一致。"""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step > cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def configure_optimizer(model, cfg):
    """weight decay 只加在 2 维以上的权重上。

    LayerNorm 的 gain、所有 bias、embedding 之外的一维参数都不衰减 ——
    这是 GPT-2 之后所有实现的默认做法，照抄就行。
    """
    decay, no_decay, seen = [], [], set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused 版在 CUDA 上快一些，老版本 torch 没有就退回普通版
    kw = {}
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, **kw)


# ---------------------------------------------------------------- 数据源

def as_batch_fn(src):
    """把 DataLoader / 可迭代对象 / 无参函数，统一成一个 () -> batch 的函数。"""
    if src is None:
        return None
    if callable(src) and not hasattr(src, "__iter__"):
        return src
    it = iter(src)

    def f():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(src)          # 数据跑完就从头再来
            return next(it)

    return f


# ---------------------------------------------------------------- Trainer

class Trainer:
    def __init__(self, model, loss_fn, train_data, val_data=None, cfg=None):
        self.cfg = cfg or TrainConfig()
        self.device = pick_device(self.cfg.device)
        set_seed(self.cfg.seed)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.train_batch = as_batch_fn(train_data)
        self.val_batch = as_batch_fn(val_data)

        self.opt = configure_optimizer(self.model, self.cfg)
        self.ema = EMA(self.model, self.cfg.ema_decay) if self.cfg.ema_decay > 0 else None

        # fp16 才需要 GradScaler；bf16 的动态范围和 fp32 一样，不需要
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.cfg.dtype == "fp16"
                                                            and self.device.type == "cuda"))
        if self.cfg.compile:
            self.model = torch.compile(self.model)

        self.step = 0
        self.history = []                 # [(step, train_loss, val_loss or None)]
        self._meter = AvgMeter()

    # -------------------------------------------------- 一步

    def _forward(self, batch):
        batch = to_device(batch, self.device)
        with autocast_ctx(self.device, self.cfg.dtype):
            out = self.loss_fn(self.model, batch)
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, {}

    def train_step(self):
        cfg = self.cfg
        self.model.train()

        for g in self.opt.param_groups:      # 每一步都重设学习率
            g["lr"] = get_lr(self.step, cfg)

        self.opt.zero_grad(set_to_none=True)
        total, metrics = 0.0, {}
        for _ in range(cfg.grad_accum):
            loss, m = self._forward(self.train_batch())
            # 每个 micro-batch 内部已经取过平均，再除以份数才等于大 batch 的平均
            self.scaler.scale(loss / cfg.grad_accum).backward()
            total += loss.item() / cfg.grad_accum
            for k, v in m.items():
                metrics[k] = metrics.get(k, 0.0) + float(v) / cfg.grad_accum

        gnorm = 0.0
        if cfg.grad_clip > 0:
            self.scaler.unscale_(self.opt)   # 裁剪前必须先把梯度还原成真实尺度
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip))

        self.scaler.step(self.opt)
        self.scaler.update()
        if self.ema is not None:
            self.ema.update(self.model)

        self.step += 1
        return total, gnorm, metrics

    # -------------------------------------------------- 评估

    @torch.no_grad()
    def evaluate(self, n_steps=None):
        if self.val_batch is None:
            return None
        n = n_steps or self.cfg.eval_steps
        self.model.eval()
        tot = 0.0
        for _ in range(n):
            loss, _ = self._forward(self.val_batch())
            tot += loss.item()
        self.model.train()
        return tot / n

    # -------------------------------------------------- 主循环

    def fit(self, max_steps=None):
        cfg = self.cfg
        steps = max_steps or cfg.max_steps
        describe(self.model, cfg.run_name)
        print(f"[{cfg.run_name}] device={self.device} dtype={cfg.dtype} "
              f"steps={steps} accum={cfg.grad_accum}")

        t0 = time.perf_counter()
        last_log = t0
        while self.step < steps:
            loss, gnorm, metrics = self.train_step()
            self._meter.update(loss)

            if cfg.log_every and self.step % cfg.log_every == 0:
                now = time.perf_counter()
                dt = (now - last_log) / cfg.log_every
                last_log = now
                extra = " ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                print(f"step {self.step:6d} | loss {self._meter.value:.4f} | "
                      f"lr {get_lr(self.step, cfg):.2e} | gn {gnorm:5.2f} | "
                      f"{dt * 1000:6.1f}ms/it | {cuda_mem():.2f}GB"
                      + (f" | {extra}" if extra else ""))

            if cfg.eval_every and self.step % cfg.eval_every == 0:
                vl = self.evaluate()
                self.history.append((self.step, self._meter.value, vl))
                if vl is not None:
                    print(f"{'':6}   -> val loss {vl:.4f}")

            if cfg.ckpt_every and self.step % cfg.ckpt_every == 0:
                self.save()

        total_dt = time.perf_counter() - t0
        print(f"[{cfg.run_name}] 训练完成 {self.step} 步，用时 {total_dt:.1f}s "
              f"({total_dt / max(1, self.step) * 1000:.1f}ms/it)")
        self.save()
        return self.history

    # -------------------------------------------------- checkpoint

    def save(self, path=None):
        p = path or ckpt_path(self.cfg.out_dir, self.cfg.run_name, self.step)
        model = getattr(self.model, "_orig_mod", self.model)   # 解开 torch.compile
        save_ckpt(p, model, self.opt, self.step, self.cfg, self.ema)
        print(f"[ckpt] {p}")
        return p

    def load(self, path):
        model = getattr(self.model, "_orig_mod", self.model)
        self.step = load_ckpt(path, model, self.opt, self.ema, map_location=self.device)
        print(f"[ckpt] 从 {path} 恢复到 step {self.step}")
        return self.step

    # -------------------------------------------------- 调试

    def overfit_one_batch(self, steps=300, lr=None, verbose=True):
        """全世界最强的调试手段：拿一个 batch 反复训，看 loss 能不能压到 ~0。

        压不下去 = 模型 / loss / 数据对齐写错了。
        这时候去调超参、换优化器、加数据全都是浪费时间，回来改代码。
        """
        batch = self.train_batch()
        old_lr, old_clip = self.cfg.lr, self.cfg.grad_clip
        self.cfg.lr = lr or max(self.cfg.lr, 1e-3)
        losses = []
        self.model.train()
        for i in range(steps):
            for g in self.opt.param_groups:
                g["lr"] = self.cfg.lr
            self.opt.zero_grad(set_to_none=True)
            loss, _ = self._forward(batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            losses.append(loss.item())
            if verbose and (i + 1) % max(1, steps // 5) == 0:
                print(f"  overfit {i + 1:4d}/{steps}  loss {losses[-1]:.6f}")
        self.cfg.lr, self.cfg.grad_clip = old_lr, old_clip
        ok = losses[-1] < 0.05 * max(losses[0], 1e-8) or losses[-1] < 1e-3
        print(f"[overfit] {losses[0]:.4f} -> {losses[-1]:.6f}  "
              + ("看起来没问题" if ok else "**压不下去，先回去查代码，别调参**"))
        return losses


# ---------------------------------------------------------------- 自检

if __name__ == "__main__":
    """跑一遍确认 Trainer 在你机器上是好的：
        python common/trainer.py
    合成数据上的三分类，10 秒内应该 loss 掉到 0.1 以下。
    """
    import torch.nn.functional as F

    set_seed(0)
    dev = pick_device("cuda")
    W = torch.randn(16, 3, device=dev) * 2

    def make_batch():
        x = torch.randn(256, 16, device=dev)
        y = (x @ W).argmax(-1)
        return x, y

    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        return loss, {"acc": acc}          # 第二项会自动进日志

    model = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Linear(128, 3))
    cfg = TrainConfig(run_name="selftest", max_steps=400, lr=3e-3, min_lr=3e-4,
                      warmup_steps=40, eval_every=100, log_every=50, ema_decay=0.99)

    tr = Trainer(model, loss_fn, make_batch, make_batch, cfg)

    print("\n--- 先做 overfit 单 batch 自检 ---")
    tr.overfit_one_batch(steps=200)

    print("\n--- 正式训练 ---")
    tr.fit()

    print("\n--- EMA 权重下的评估 ---")
    with tr.ema.swapped(tr.model):
        print(f"  val loss (EMA) = {tr.evaluate(20):.4f}")
