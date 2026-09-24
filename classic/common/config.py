"""训练配置。

所有开源训练项目的入口都长这样：一个 dataclass + 一个 CLI 解析。
Day 1 你自己写过一遍，这里是加长版，后面 12 天直接改字段就行。
"""
from dataclasses import dataclass, field, asdict, fields
from typing import Tuple


@dataclass
class TrainConfig:
    # ---- 运行 ----
    run_name: str = "run"
    out_dir: str = "ckpt"
    seed: int = 0
    device: str = "cuda"          # "cuda" / "cpu"，auto 会在 Trainer 里兜底
    dtype: str = "bf16"           # "bf16" | "fp16" | "fp32"；5080 上用 bf16
    compile: bool = False         # torch.compile，第一次编译要等 30 秒左右

    # ---- 优化 ----
    max_steps: int = 1000
    batch_size: int = 32          # Trainer 不用它，是给你造 dataloader 时用的
    grad_accum: int = 1           # 显存不够时把它调大、batch_size 调小
    lr: float = 3e-4
    min_lr: float = 3e-5          # cosine 衰减的下界
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0        # 0 表示不裁剪

    # ---- 评估与日志 ----
    eval_every: int = 200         # 0 表示不评估
    eval_steps: int = 20          # 每次评估跑几个 batch
    log_every: int = 20
    ckpt_every: int = 0           # 0 表示只在训练结束时存一次

    # ---- EMA（扩散模型基本都要开）----
    ema_decay: float = 0.0        # 0 表示关闭，扩散常用 0.999

    # ---- 杂项 ----
    tags: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        w = max(len(f.name) for f in fields(self))
        lines = [f"  {f.name:<{w}} = {getattr(self, f.name)!r}" for f in fields(self)]
        return "TrainConfig(\n" + "\n".join(lines) + "\n)"


def cli(cfg_cls=TrainConfig, **overrides):
    """从命令行覆盖任意字段：

        python train.py --lr 1e-3 --max_steps 500 --compile

    bool 字段写成 --compile 就是 True，--no-compile 就是 False。
    """
    import argparse

    base = cfg_cls(**overrides)
    p = argparse.ArgumentParser()
    for f in fields(base):
        cur = getattr(base, f.name)
        if isinstance(cur, bool):
            p.add_argument(f"--{f.name}", dest=f.name, action="store_true", default=None)
            p.add_argument(f"--no-{f.name}", dest=f.name, action="store_false")
        elif isinstance(cur, (tuple, list)):
            continue                      # 元组/列表走代码里改，别从命令行传
        else:
            p.add_argument(f"--{f.name}", type=type(cur), default=None)
    args = p.parse_args()
    for k, v in vars(args).items():
        if v is not None:
            setattr(base, k, v)
    return base
