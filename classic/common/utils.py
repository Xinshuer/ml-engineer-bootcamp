"""小工具。没有一个超过 20 行，但后面 12 天每天都用。"""
import contextlib
import math
import os
import random
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------- 基础

def set_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:                      # 慢，但能完全复现，查 bug 时打开
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(want="cuda"):
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def human(n):
    """1234567 -> '1.23M'"""
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def count_params(model):
    """返回 (总数, 可训练数)。共享权重只数一遍。"""
    seen, total, train = set(), 0, 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            train += p.numel()
    return total, train


def describe(model, name="model"):
    total, train = count_params(model)
    pct = 100 * train / total if total else 0
    print(f"[{name}] 参数 {human(total)}  可训练 {human(train)} ({pct:.1f}%)")
    return total, train


# ---------------------------------------------------------------- 设备与 dtype

def to_device(batch, device, non_blocking=True):
    """递归把 tensor / tuple / list / dict 里的张量搬到 device 上。"""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, (tuple, list)):
        return type(batch)(to_device(b, device, non_blocking) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device, non_blocking) for k, v in batch.items()}
    return batch


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def autocast_ctx(device, dtype_str):
    """bf16 不需要 GradScaler，fp16 需要。fp32 直接返回空上下文。"""
    if device.type != "cuda" or dtype_str == "fp32":
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=DTYPES[dtype_str])


# ---------------------------------------------------------------- 统计

class AvgMeter:
    """滑动平均，用来平滑抖动的 loss 曲线。"""

    def __init__(self, momentum=0.9):
        self.m = momentum
        self.v = None
        self.last = 0.0

    def update(self, x):
        x = float(x)
        self.last = x
        self.v = x if self.v is None else self.m * self.v + (1 - self.m) * x
        return self.v

    @property
    def value(self):
        return self.v if self.v is not None else 0.0


class Timer:
    """with Timer('前向') as t: ...   然后 t.dt 是秒数。"""

    def __init__(self, name=None, sync=True):
        self.name, self.sync = name, sync
        self.dt = 0.0

    def __enter__(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.t0
        if self.name:
            print(f"[{self.name}] {self.dt * 1000:.1f}ms")


def cuda_mem():
    """当前显存占用（GB）。OOM 时用它定位是哪一步涨的。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


# ---------------------------------------------------------------- EMA

class EMA:
    """指数滑动平均。扩散模型不开这个，采样结果会明显更糊。

        ema = EMA(model, 0.999)
        ...每步 optimizer.step() 之后:  ema.update(model)
        采样时:  with ema.swapped(model): sample(model)
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    @contextlib.contextmanager
    def swapped(self, model):
        """临时把 EMA 权重换进模型，退出时换回来。"""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({k: v.to(dtype=backup[k].dtype) for k, v in self.shadow.items()},
                              strict=False)
        try:
            yield model
        finally:
            model.load_state_dict(backup, strict=False)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


# ---------------------------------------------------------------- checkpoint

def ckpt_path(out_dir, run_name, step):
    d = os.path.join(out_dir, run_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"ckpt_{step:07d}.pt")


def save_ckpt(path, model, optimizer=None, step=0, cfg=None, ema=None, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    obj = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if cfg is not None:
        obj["cfg"] = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
    if ema is not None:
        obj["ema"] = ema.state_dict()
    obj.update(extra)
    torch.save(obj, path)
    return path


def load_ckpt(path, model, optimizer=None, ema=None, map_location="cpu"):
    obj = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(obj["model"])
    if optimizer is not None and "optimizer" in obj:
        optimizer.load_state_dict(obj["optimizer"])
    if ema is not None and "ema" in obj:
        ema.load_state_dict(obj["ema"])
    return obj.get("step", 0)


# ---------------------------------------------------------------- 调试

@torch.no_grad()
def print_shapes(model, *inputs, max_lines=40):
    """给每个子模块挂 hook，打印一次前向里所有中间张量的 shape。
    实现一个新架构时，这个比 print 大法快十倍。
    """
    lines = []

    def hook(name):
        def f(_mod, _inp, out):
            if torch.is_tensor(out):
                lines.append(f"  {name:<40} {tuple(out.shape)}")
        return f

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]
    try:
        model(*inputs)
    finally:
        for h in handles:
            h.remove()
    print("\n".join(lines[:max_lines]))
    if len(lines) > max_lines:
        print(f"  ... 还有 {len(lines) - max_lines} 行")


def smoke_test(model, *input_shapes, expect_shape=None, device="cpu"):
    """写完 model.py 的第一件事：喂随机张量跑一次，确认 shape 和数值都正常。"""
    model = model.to(device)
    xs = [torch.randn(s, device=device) for s in input_shapes]
    out = model(*xs)
    ref = out[0] if isinstance(out, (tuple, list)) else out
    if expect_shape is not None and tuple(ref.shape) != tuple(expect_shape):
        raise AssertionError(f"输出 shape {tuple(ref.shape)} != 期望 {tuple(expect_shape)}")
    if not torch.isfinite(ref).all():
        raise AssertionError("输出里有 nan / inf")
    print(f"[smoke] OK  输入 {input_shapes} -> 输出 {tuple(ref.shape)}")
    return True
