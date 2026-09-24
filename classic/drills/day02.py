"""Day 02 · PyTorch 的第一语言是 shape —— 15 题

运行:  python drills/day02.py
这一天是全课地基。第 1 题请**用脑子算**，不许先跑代码。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, params

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 01  shape 速算 —— 不许运行，先在脑子里算完再填
# ==================================================================
# 给定:
#   a: (8, 16, 64)      b: (64, 32)       c: (8, 1, 64)
#   d: (4, 6, 8, 8)     e: (64,)
# 把每个表达式的结果 shape 填进下面的 dict（用 tuple）。
# 算错了不丢人，跑一遍看看差在哪，这一题的价值全在"先猜后验"。

SHAPES = {
    "a @ b":                        None,   # 矩阵乘，最后一维对齐
    "a + c":                        None,   # 广播
    "a.transpose(0, 1)":            None,
    "a.view(8, 16, 4, 16)":         None,
    "a.view(8, 16, 4, 16).transpose(1, 2)": None,
    "a.mean(dim=-1)":               None,
    "a.mean(dim=-1, keepdim=True)": None,
    "d.permute(0, 2, 1, 3)":        None,
    "d.flatten(2)":                 None,
    "a * e":                        None,   # 广播从右往左对齐，想清楚为什么 e 必须是 (64,)
    "c + torch.zeros(16, 1)":       None,   # 两边都要广播
    "a.unsqueeze(1)":               None,
    "torch.cat([a, a], dim=-1)":    None,
}


@task(1, "shape 速算 (先脑算再跑)")
def t01():
    a = torch.zeros(8, 16, 64)
    b = torch.zeros(64, 32)
    c = torch.zeros(8, 1, 64)
    d = torch.zeros(4, 6, 8, 8)
    e = torch.zeros(64)
    env = dict(a=a, b=b, c=c, d=d, e=e, torch=torch)
    wrong = []
    for expr, guess in SHAPES.items():
        real = tuple(eval(expr, env).shape)
        if guess is None:
            wrong.append(f"{expr!r} 还没填")
        elif tuple(guess) != real:
            wrong.append(f"{expr!r}: 你填 {tuple(guess)}, 实际 {real}")
    true(not wrong, "\n       " + "\n       ".join(wrong))


# ==================================================================
# 02  qkv 拆头 —— 明天写 attention 时一模一样的代码
# ==================================================================
def split_heads(qkv, n_head):
    """qkv: (B, T, 3*C)  ->  三个 (B, n_head, T, head_dim) 的张量 q, k, v
    顺序是先 q 后 k 再 v（沿最后一维等分三份）。
    """
    # --- TODO 02 ---
    raise todo()


@task(2, "qkv 拆头 (B,T,3C) -> 3 x (B,nh,T,hs)")
def t02():
    B, T, C, nh = 2, 5, 12, 3
    qkv = torch.randn(B, T, 3 * C)
    q, k, v = split_heads(qkv, nh)
    for name, t in [("q", q), ("k", k), ("v", v)]:
        shape_is(t, (B, nh, T, C // nh), f"({name})")
    # 内容也要对: q 必须来自 qkv 的前 C 列
    q_ref = qkv[..., :C].view(B, T, nh, C // nh).transpose(1, 2)
    eq(q, q_ref, msg="q 的内容不对 —— 检查 view 和 transpose 的顺序")


# ==================================================================
# 03  contiguous 的那个经典报错
# ==================================================================
def flatten_heads(x):
    """x: (B, n_head, T, head_dim)  ->  (B, T, n_head*head_dim)
    这是 attention 输出合并回去的那一步。直接 view 会报错，想清楚为什么。
    """
    # --- TODO 03 ---
    raise todo()


@task(3, "合并头 + contiguous")
def t03():
    B, nh, T, hs = 2, 3, 5, 4
    x = torch.randn(B, nh, T, hs)
    y = flatten_heads(x)
    shape_is(y, (B, T, nh * hs))
    eq(y[0, 1], torch.cat([x[0, h, 1] for h in range(nh)]),
       msg="元素排列不对 —— 同一个时间步的各个头要拼在一起")


# ==================================================================
# 04  手写 Linear，和 nn.Linear 对拍
# ==================================================================
def my_linear(x, W, b):
    """x: (..., in)   W: (out, in)   b: (out,)   ->  (..., out)
    注意 nn.Linear 的权重是 (out_features, in_features)，不是 (in, out)。
    """
    # --- TODO 04 ---
    raise todo()


@task(4, "手写 Linear 对拍 nn.Linear")
def t04():
    seed(0)
    lin = nn.Linear(7, 3)
    x = torch.randn(2, 5, 7)
    eq(my_linear(x, lin.weight, lin.bias), lin(x))


# ==================================================================
# 05  数值稳定的 softmax
# ==================================================================
def my_softmax(x, dim=-1):
    """不许用 F.softmax / x.softmax。必须减去最大值，否则大数会溢出。"""
    # --- TODO 05 ---
    raise todo()


@task(5, "数值稳定 softmax")
def t05():
    x = torch.randn(3, 4, 5)
    eq(my_softmax(x, -1), F.softmax(x, -1))
    eq(my_softmax(x, 1), F.softmax(x, 1), msg="dim 参数没用上")
    big = torch.tensor([[1000.0, 1001.0, 1002.0]])
    out = my_softmax(big, -1)
    true(torch.isfinite(out).all(), "大数溢出了 —— 忘了减最大值")
    eq(out, F.softmax(big, -1))


# ==================================================================
# 06  手写 cross_entropy —— 训 LLM 每一步都在算这个
# ==================================================================
def my_cross_entropy(logits, target):
    """logits: (N, V) 未归一化；target: (N,) int64 类别下标。
    返回标量（对 N 取平均）。不许用 F.cross_entropy / F.nll_loss。
    """
    # --- TODO 06 ---
    raise todo()


@task(6, "手写 cross_entropy")
def t06():
    seed(1)
    logits = torch.randn(6, 10)
    y = torch.randint(0, 10, (6,))
    eq(my_cross_entropy(logits, y), F.cross_entropy(logits, y))
    # 均匀分布时 loss 应该是 ln(V)
    flat = torch.zeros(4, 10)
    eq(my_cross_entropy(flat, torch.zeros(4, dtype=torch.long)), torch.tensor(math.log(10)),
       msg="均匀 logits 的 loss 应该正好是 ln(V) —— 这个数字你要背下来")


# ==================================================================
# 07  展平 (B,T,V) 去算 loss —— 这里错一次能查一整天
# ==================================================================
def seq_loss(logits, targets):
    """logits: (B, T, V)   targets: (B, T) int64  -> 标量 loss
    F.cross_entropy 只吃二维，所以要先展平。允许用 F.cross_entropy。
    """
    # --- TODO 07 ---
    raise todo()


@task(7, "序列 loss 的展平")
def t07():
    seed(2)
    B, T, V = 3, 4, 7
    logits = torch.randn(B, T, V)
    y = torch.randint(0, V, (B, T))
    ref = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    eq(seq_loss(logits, y), ref)
    # 反例检查: 如果你写成 logits.view(-1, V) 但 y 用了 y.view(V, -1) 之类，下面会挂
    true(seq_loss(logits, y).dim() == 0, "loss 应该是标量")


# ==================================================================
# 08  手写 LayerNorm 与 RMSNorm
# ==================================================================
def my_layernorm(x, g, b, eps=1e-5):
    """在最后一维上归一化。g, b 形状都是 (C,)。"""
    # --- TODO 08 ---
    raise todo()


def my_rmsnorm(x, g, eps=1e-6):
    """RMSNorm: 不减均值，只除以 sqrt(mean(x^2) + eps)，再乘 g。Day 5 要用。"""
    # --- TODO 08b ---
    raise todo()


@task(8, "LayerNorm / RMSNorm")
def t08():
    seed(3)
    x = torch.randn(2, 5, 8)
    g, b = torch.randn(8), torch.randn(8)
    eq(my_layernorm(x, g, b), F.layer_norm(x, (8,), g, b))
    r = my_rmsnorm(x, torch.ones(8))
    eq(r.pow(2).mean(-1), torch.ones(2, 5), tol=1e-3,
       msg="RMSNorm 之后每个向量的均方值应该 ≈ 1")


# ==================================================================
# 09  einsum：同一个操作的三种写法
# ==================================================================
def attn_scores_matmul(q, k):
    """q, k: (B, nh, T, hs) -> (B, nh, T, T)，用 @ / matmul 写，别忘了除以 sqrt(hs)"""
    # --- TODO 09a ---
    raise todo()


def attn_scores_einsum(q, k):
    """同上，但必须用 torch.einsum"""
    # --- TODO 09b ---
    raise todo()


@task(9, "attention scores: matmul vs einsum")
def t09():
    seed(4)
    q, k = torch.randn(2, 3, 5, 4), torch.randn(2, 3, 5, 4)
    a, b = attn_scores_matmul(q, k), attn_scores_einsum(q, k)
    shape_is(a, (2, 3, 5, 5))
    eq(a, b, msg="两种写法结果应该完全一致")
    eq(a[0, 0, 1, 2], (q[0, 0, 1] * k[0, 0, 2]).sum() / math.sqrt(4),
       msg="缩放因子 1/sqrt(head_dim) 没加或加错了")


# ==================================================================
# 10  masked 统计 —— 变长序列天天用
# ==================================================================
def masked_mean(x, mask):
    """x: (B, T, C)   mask: (B, T) 的 0/1（1 表示有效）
    对每个样本在有效位置上求 C 维向量的平均 -> (B, C)。
    不许用循环。假设每个样本至少有一个有效位。
    """
    # --- TODO 10 ---
    raise todo()


@task(10, "masked_mean")
def t10():
    x = torch.arange(2 * 3 * 2, dtype=torch.float).reshape(2, 3, 2)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    got = masked_mean(x, mask)
    want = torch.stack([x[0, :2].mean(0), x[1, 1:].mean(0)])
    eq(got, want, msg="被 mask 掉的位置不能参与平均，分母也要跟着变")


# ==================================================================
# 11  gather —— 取出每个位置"正确答案"的 logit
# ==================================================================
def pick_target_logits(logits, targets):
    """logits: (B, T, V)   targets: (B, T)  ->  (B, T)
    取出每个 (b,t) 位置上 targets[b,t] 对应的那个 logit。用 torch.gather。
    """
    # --- TODO 11 ---
    raise todo()


@task(11, "torch.gather")
def t11():
    seed(5)
    logits = torch.randn(2, 3, 6)
    y = torch.randint(0, 6, (2, 3))
    got = pick_target_logits(logits, y)
    shape_is(got, (2, 3))
    want = torch.stack([torch.stack([logits[b, t, y[b, t]] for t in range(3)]) for b in range(2)])
    eq(got, want)


# ==================================================================
# 12  nn.Module: Parameter vs Buffer
# ==================================================================
class Scaler(nn.Module):
    """要求:
      - 可学习参数 weight，形状 (dim,)，初始化为全 1
      - 不可学习但要跟着 state_dict 走 / 跟着 .to(device) 走的 buffer: running_max，形状 (dim,)，初始 0
      - forward(x): 更新 running_max = max(running_max, x 在 batch 维的最大值)（不参与梯度），
        返回 x * weight
    """

    def __init__(self, dim):
        super().__init__()
        # --- TODO 12 ---
        raise todo()


@task(12, "Parameter vs Buffer")
def t12():
    m = Scaler(4)
    keys = set(m.state_dict().keys())
    true(keys == {"weight", "running_max"}, f"state_dict 的 key 不对: {keys}")
    true([n for n, _ in m.named_parameters()] == ["weight"],
         "running_max 不应该是 Parameter（它不该被优化器更新）")
    x = torch.randn(8, 4)
    out = m(x)
    eq(out, x)
    eq(m.running_max, x.max(0).values.clamp(min=0))
    true(not m.running_max.requires_grad, "buffer 不该需要梯度")


# ==================================================================
# 13  autograd：手推梯度 vs backward 对拍
# ==================================================================
def manual_grad_mse(x, target):
    """y = mean((x - target)^2)，手写 dy/dx 的解析式并返回（不许用 backward）。"""
    # --- TODO 13 ---
    raise todo()


@task(13, "手推梯度对拍 autograd")
def t13():
    seed(6)
    x = torch.randn(3, 4, requires_grad=True)
    t = torch.randn(3, 4)
    loss = ((x - t) ** 2).mean()
    loss.backward()
    eq(manual_grad_mse(x.detach(), t), x.grad,
       msg="别忘了 mean 会带来一个 1/N 的系数")


# ==================================================================
# 14  collate_fn：把不等长的样本 pad 成一个 batch
# ==================================================================
def collate(samples, pad_id=0):
    """samples: list of 1D LongTensor，长度不一。
    返回 (x, mask):
      x    : (B, Tmax) int64，短的右侧补 pad_id
      mask : (B, Tmax) int64，有效位置 1，pad 位置 0
    """
    # --- TODO 14 ---
    raise todo()


@task(14, "collate_fn 变长 padding")
def t14():
    s = [torch.tensor([1, 2, 3]), torch.tensor([4]), torch.tensor([5, 6])]
    x, mask = collate(s, pad_id=99)
    shape_is(x, (3, 3))
    eq(x, torch.tensor([[1, 2, 3], [4, 99, 99], [5, 6, 99]]))
    eq(mask, torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]]))
    true(x.dtype == torch.long, "dtype 必须是 int64，embedding 层只吃这个")


# ==================================================================
# 15  找 bug —— 这个训练循环有 4 处错，改到 loss 真的下降为止
# ==================================================================
def train_one_epoch(model, xs, ys, lr=0.1, steps=200):
    opt = torch.optim.SGD(model.parameters, lr=lr)
    losses = []
    for _ in range(steps):
        logits = model(xs)
        loss = F.cross_entropy(logits, ys.float())
        loss.backward()
        opt.step()
        losses.append(loss)
    return losses


@task(15, "找 bug: 训练循环")
def t15():
    seed(7)
    model = nn.Linear(5, 3)
    xs, ys = torch.randn(64, 5), torch.randint(0, 3, (64,))
    losses = train_one_epoch(model, xs, ys)
    true(isinstance(losses[0], float),
         "losses 里存的是 tensor，会把整张计算图留在显存里 —— 要存 .item()")
    true(losses[-1] < losses[0] * 0.9,
         f"loss 没降下来: {losses[0]:.4f} -> {losses[-1]:.4f}。少了哪一步？")


if __name__ == "__main__":
    raise SystemExit(run("Day 02 · 张量与 shape"))
