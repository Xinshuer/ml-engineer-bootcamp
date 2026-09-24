"""Day 01 参考答案。卡住超过 5 分钟再看。"""
from dataclasses import dataclass, field
from pathlib import Path
import functools

CALLS = []


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 32
    n_layer: int = 6
    device: str = "cuda"
    betas: tuple = (0.9, 0.95)
    tags: list = field(default_factory=list)   # 可变默认值必须走 default_factory

    def scaled_lr(self, k):
        return self.lr * k


def chunk(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq) - n + 1, n)]


def make_xy(data, i, block_size):
    x = data[i:i + block_size]
    y = data[i + 1:i + 1 + block_size]
    return x, y


def char_freq(text):
    f = {}
    for ch in text:
        f[ch] = f.get(ch, 0) + 1
    return f


def top_k(freq, k):
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


class RunningMean:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.n = 0

    def __call__(self, x):
        self.total += float(x)
        self.n += 1
        return self.value

    @property
    def value(self):
        return self.total / self.n if self.n else 0.0


class EMAMean(RunningMean):
    def __init__(self, decay=0.9):
        super().__init__()
        self.decay = decay
        self._v = None

    def __call__(self, x):
        x = float(x)
        self._v = x if self._v is None else self.decay * self._v + (1 - self.decay) * x
        self.n += 1
        return self._v

    @property
    def value(self):
        return 0.0 if self._v is None else self._v

    def reset(self):
        super().reset()
        self._v = None


class Budget:
    def __init__(self, total_steps, step=0):
        self.total_steps = total_steps
        self.step = step

    @property
    def progress(self):
        return self.step / self.total_steps

    @property
    def done(self):
        return self.step >= self.total_steps


def logged(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        CALLS.append((fn.__name__, args, kwargs))
        return fn(*args, **kwargs)
    return wrapper


def log_line(step, loss, lr, dt):
    return f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | {dt * 1000:.1f}ms"


def deltas(xs):
    return [b - a for a, b in zip(xs, xs[1:])]


def label_pairs(names, values):
    return [f"{i}:{n}={v}" for i, (n, v) in enumerate(zip(names, values))]


def ckpt_path(root, run_name, step):
    d = Path(root) / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ckpt_{step:06d}.pt"


def bucket_by_len(samples, bucket_size=2):
    out = {}                       # 修 1: 可变默认值挪进函数体
    for s in samples:
        L = len(s)
        out.setdefault(L, [])
        if len(out[L]) >= bucket_size:
            continue               # 修 2+3: 满了跳过，不 append 也不报错
        out[L].append(s)
    return out
