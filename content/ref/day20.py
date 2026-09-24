"""Day 20 参考代码：checkpoint 的坑。

题目把这个文件当作隐藏的准备代码加载，然后把题目要你写的名字（targets）删掉。
里面有：小模型 Net、共享权重的 TinyLM、模仿 DDP 的 Wrapper、训练三件套 make_run / train_step，
几道题的参考答案，以及检查「续训 == 没中断」用的实验 _resume_run。
"""
import copy
import os
import random
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, load_model, safe_open, save_file, save_model


class Net(nn.Module):
    """小模型：body -> relu -> dropout -> head。
    dropout 每次随机丢掉一些值，所以「随机数状态」也是训练状态的一部分。"""

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 8)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.drop(F.relu(self.body(x))))


class TinyLM(nn.Module):
    """词嵌入和输出层共用同一个权重（tied weights），GPT-2 就是这样做的。"""

    def __init__(self, vocab=10, dim=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.emb.weight          # 两个名字，同一个 Parameter

    def forward(self, idx):
        return self.head(self.emb(idx))


class Wrapper(nn.Module):
    """模仿 DDP / DataParallel：把模型放在 self.module 里，所以 state_dict 的 key 都多了 'module.'。"""

    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *args):
        return self.module(*args)


def make_run():
    """一次训练需要的三件套：模型、优化器（AdamW）、学习率调度器（每 2 步学习率减半）。"""
    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    return model, opt, sched


def train_step(model, opt, sched):
    """训练一步，返回 loss（float）。数据和 dropout 都用全局随机数。"""
    model.train()
    x = torch.randn(8, 4)
    y = x[:, :2] * 2
    loss = F.mse_loss(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    return loss.item()


# ---------------------------------------------------------------- 参考答案
def strip_prefix(sd, prefixes=("module.", "_orig_mod.")):
    out = {}
    for key, value in sd.items():
        while key.startswith(prefixes):          # startswith 可以收一个 tuple：以其中任意一个开头就是 True
            for p in prefixes:
                key = key.removeprefix(p)
        out[key] = value
    return out


def load_strict(model, sd):
    want = model.state_dict()
    missing = sorted(set(want) - set(sd))
    unexpected = sorted(set(sd) - set(want))
    wrong_shape = [f"{k}: checkpoint {tuple(sd[k].shape)} vs model {tuple(want[k].shape)}"
                   for k in want if k in sd and tuple(sd[k].shape) != tuple(want[k].shape)]
    if missing or unexpected or wrong_shape:
        raise ValueError(
            "checkpoint 和模型对不上:\n"
            f"  missing（模型有、文件里没有）: {missing}\n"
            f"  unexpected（文件里有、模型没有）: {unexpected}\n"
            f"  shape 不同: {wrong_shape}")
    model.load_state_dict(sd)
    return model


def save_checkpoint(path, model, opt, sched, step):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "rng_python": random.getstate(),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, opt, sched):
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["rng_torch"])
    if ckpt["rng_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    random.setstate(ckpt["rng_python"])
    return ckpt["step"]


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---------------------------------------------------------------- 检查用的实验
def _resume_run(save_fn, load_fn):
    """证明「续训 == 没中断」。
    A：不间断训练 5 步。
    B：训练 3 步 -> save_fn 存盘 -> 打乱随机数、造全新的对象（像重启了进程）-> load_fn 加载 -> 再训练 2 步。
    返回一个 dict，检查项从里面取自己要看的东西。"""
    torch.manual_seed(0)
    model, opt, sched = make_run()
    full = [train_step(model, opt, sched) for _ in range(5)]
    full_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.manual_seed(0)
    model, opt, sched = make_run()
    first = [train_step(model, opt, sched) for _ in range(3)]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        path = os.path.join(folder, "ckpt.pt")
        save_fn(path, model, opt, sched, 3)
        saved = {
            "model": {k: v.clone() for k, v in model.state_dict().items()},
            "opt": copy.deepcopy(opt.state_dict()),
            "lr": opt.param_groups[0]["lr"],
            "last_epoch": sched.last_epoch,
            "rng": torch.get_rng_state().clone(),
        }
        torch.manual_seed(1234)              # 「重启进程」：随机数状态和刚才完全不同
        model2, opt2, sched2 = make_run()    # 全新的对象，权重也是新的随机初始化
        step = load_fn(path, model2, opt2, sched2)
        loaded = {
            "model": {k: v.clone() for k, v in model2.state_dict().items()},
            "opt": copy.deepcopy(opt2.state_dict()),
            "lr": opt2.param_groups[0]["lr"],
            "last_epoch": sched2.last_epoch,
            "rng": torch.get_rng_state().clone(),
            "step": step,
        }
        rng_now = torch.get_rng_state()
        try:
            file, file_error = torch.load(path, weights_only=True), None
        except Exception as e:  # noqa: BLE001 - the check reports it
            file, file_error = None, e
        torch.set_rng_state(rng_now)
    rest = [train_step(model2, opt2, sched2) for _ in range(2)]
    return dict(full=full, full_state=full_state, first=first, rest=rest,
                resumed_state={k: v.clone() for k, v in model2.state_dict().items()},
                saved=saved, loaded=loaded, file=file, file_error=file_error)
