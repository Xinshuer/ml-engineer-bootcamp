"""自测小工具。所有 drills/dayNN.py 都靠它跑。

用法（在 ai-14days/ 目录下）:
    python drills/day02.py          # 跑 day02 全部题
    python drills/day02.py 05       # 只跑第 5 题
    python check.py 02              # 同上，等价入口
    python check.py 02 --sol        # 用参考答案跑一遍（验证题目本身没问题）
"""
import importlib
import os
import sys
import traceback

import torch

# 让 `import solutions.dayNN` 能找到
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:  # Windows 控制台默认 cp936，中文能过但符号会炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_TASKS = []


def task(tid, title):
    """把一个检查函数注册成一道题。"""

    def deco(fn):
        _TASKS.append((str(tid).zfill(2), title, fn, fn.__module__))
        return fn

    return deco


def todo(hint=""):
    raise NotImplementedError(hint or "这里还没写")


# ---------- 断言工具 ----------

def eq(got, want, tol=1e-4, msg=""):
    """数值对拍。shape 先查，再查最大绝对误差。"""
    g = torch.as_tensor(got)
    w = torch.as_tensor(want)
    if tuple(g.shape) != tuple(w.shape):
        raise AssertionError(f"shape 不对: 你的 {tuple(g.shape)}, 应为 {tuple(w.shape)}. {msg}")
    d = (g.float() - w.float()).abs().max().item()
    if not (d <= tol):
        raise AssertionError(f"数值不对: max|差| = {d:.3e} > 容差 {tol}. {msg}")


def shape_is(t, s, msg=""):
    got = tuple(t.shape) if hasattr(t, "shape") else tuple(t)
    if got != tuple(s):
        raise AssertionError(f"shape 不对: 你的 {got}, 应为 {tuple(s)}. {msg}")


def true(cond, msg="断言失败"):
    if not cond:
        raise AssertionError(msg)


def close(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


def seed(n=0):
    torch.manual_seed(n)


def params(m):
    return sum(p.numel() for p in m.parameters())


def trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------- 运行器 ----------

def run(title):
    import __main__ as M

    if os.environ.get("DRILL_SOL") == "1":
        name = os.path.splitext(os.path.basename(M.__file__))[0]
        sol = importlib.import_module(f"solutions.{name}")
        import types

        n = 0
        for k, v in vars(sol).items():
            if k.startswith("_") or isinstance(v, types.ModuleType):
                continue
            if k in ("task", "todo", "run", "true", "eq", "shape_is", "seed", "close",
                     "params", "trainable"):
                continue
            setattr(M, k, v)
            n += 1
        print(f"[参考答案模式] 已注入 {n} 个实现\n")

    only = None
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            only = a.zfill(2)

    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)

    ok = bad = skip = 0
    first_fail = None
    for tid, name, fn, mod in _TASKS:
        if mod != "__main__":          # 别把 import 进来的题目也跑一遍
            continue
        if only and tid != only:
            continue
        try:
            fn()
        except NotImplementedError:
            print(f"[--] {tid}  {name}")
            skip += 1
            continue
        except Exception as e:
            print(f"[XX] {tid}  {name}")
            print(f"       -> {type(e).__name__}: {e}")
            if first_fail is None:
                first_fail = (tid, e, traceback.format_exc())
            bad += 1
            continue
        print(f"[OK] {tid}  {name}")
        ok += 1

    total = ok + bad + skip
    print("-" * 64)
    print(f"  通过 {ok}/{total}    错 {bad}    没写 {skip}")
    if bad and os.environ.get("DRILL_TB") == "1":
        print("\n首个失败的完整栈:\n" + first_fail[2])
    elif bad:
        print("  想看完整报错栈: set DRILL_TB=1 再跑")
    print("=" * 64)
    return 1 if bad else 0
