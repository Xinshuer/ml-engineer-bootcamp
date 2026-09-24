"""Day 09 · 条件化：mini Stable Diffusion —— 12 题

运行:  python drills/day09.py
第 8 题（DDIM 精确还原）会让你彻底理解为什么 DDIM 是确定性的。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, close

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 昨天已经练过的工具，这里直接用参考版，免得 day08 没写完就卡在这
from solutions.day08 import linear_schedule, q_sample, extract   # noqa: E402


# ==================================================================
# 01  空间特征 <-> token 序列
# ==================================================================
def to_tokens(x):
    """(B, C, H, W) -> (B, H*W, C)。这一步之后，图像就变成了"一句话"。"""
    # --- TODO 01 ---
    raise todo()


def to_spatial(tok, H, W):
    """(B, H*W, C) -> (B, C, H, W)，是 to_tokens 的逆操作。"""
    # --- TODO 01b ---
    raise todo()


@task(1, "空间 <-> token 往返")
def t01():
    x = torch.randn(2, 8, 4, 6)
    tok = to_tokens(x)
    shape_is(tok, (2, 24, 8))
    eq(to_spatial(tok, 4, 6), x, msg="往返必须完全还原（顺序是行优先）")
    eq(tok[0, 0], x[0, :, 0, 0], msg="第 0 个 token 应该是左上角像素的通道向量")
    eq(tok[0, 1], x[0, :, 0, 1], msg="第 1 个 token 应该是它右边那个像素（行优先）")


# ==================================================================
# 02  cross-attention：Q 来自图，KV 来自文本
# ==================================================================
def cross_attention(x, ctx, wq, wk, wv, n_head):
    """x:   (B, Tx, C)    图像 token
    ctx: (B, Tc, Cc)   文本嵌入（长度和 Tx 完全无关）
    wq: (C, C)   wk: (Cc, C)   wv: (Cc, C)
    返回 (B, Tx, C)。允许用 F.scaled_dot_product_attention（不需要 causal mask）。
    """
    # --- TODO 02 ---
    raise todo()


@task(2, "cross-attention")
def t02():
    seed(0)
    B, Tx, Tc, C, Cc, nh = 2, 16, 7, 32, 64, 4
    x, ctx = torch.randn(B, Tx, C), torch.randn(B, Tc, Cc)
    wq, wk, wv = torch.randn(C, C) * .1, torch.randn(Cc, C) * .1, torch.randn(Cc, C) * .1
    out = cross_attention(x, ctx, wq, wk, wv, nh)
    shape_is(out, (B, Tx, C), "输出长度必须跟着图像走，不是跟着文本走")

    q = (x @ wq).view(B, Tx, nh, C // nh).transpose(1, 2)
    k = (ctx @ wk).view(B, Tc, nh, C // nh).transpose(1, 2)
    v = (ctx @ wv).view(B, Tc, nh, C // nh).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Tx, C)
    eq(out, ref, tol=1e-4)

    # 文本长度换一个也要能跑 —— 这正是 cross-attn 比 concat 好用的地方
    shape_is(cross_attention(x, torch.randn(B, 77, Cc), wq, wk, wv, nh), (B, Tx, C))


# ==================================================================
# 03  条件嵌入的注入：最省事的那种
# ==================================================================
def combine_cond(t_emb, class_emb):
    """class-conditional 最简单的做法: 直接把类别嵌入加到时间嵌入上。
    两个都是 (B, D)。返回 (B, D)。
    """
    # --- TODO 03 ---
    raise todo()


@task(3, "条件嵌入相加")
def t03():
    a, b = torch.randn(4, 16), torch.randn(4, 16)
    eq(combine_cond(a, b), a + b)


# ==================================================================
# 04  训练时的条件 dropout —— CFG 的前提
# ==================================================================
def drop_cond(cond_emb, null_emb, p, generator=None):
    """以概率 p 把某些样本的条件整条换成 null_emb（无条件嵌入）。
    cond_emb: (B, D)   null_emb: (D,)   返回 (B, D)。
    必须是**按样本**丢，不是按元素丢。
    """
    # --- TODO 04 ---
    raise todo()


@task(4, "条件 dropout")
def t04():
    seed(1)
    B, D = 4000, 8
    cond = torch.ones(B, D)
    null = torch.zeros(D)
    out = drop_cond(cond, null, p=0.1)
    shape_is(out, (B, D))
    per_row = out.sum(1)
    true(set(per_row.unique().tolist()) <= {0.0, float(D)},
         "有的行被丢了一半 —— 必须整行丢弃，不能按元素")
    frac = (per_row == 0).float().mean().item()
    true(abs(frac - 0.1) < 0.02, f"丢弃比例应该 ≈ 0.1, 实际 {frac:.3f}")
    eq(drop_cond(cond, null, p=0.0), cond, msg="p=0 时不该丢")
    eq(drop_cond(cond, null, p=1.0), torch.zeros(B, D), msg="p=1 时应该全丢")


# ==================================================================
# 05  CFG 的组合公式
# ==================================================================
def cfg_combine(eps_uncond, eps_cond, scale):
    """eps = eps_uncond + scale * (eps_cond - eps_uncond)
    scale=1 时就是普通条件生成；scale>1 时"往条件方向多推一点"。
    """
    # --- TODO 05 ---
    raise todo()


@task(5, "CFG 组合公式")
def t05():
    u, c = torch.randn(2, 3), torch.randn(2, 3)
    eq(cfg_combine(u, c, 1.0), c, msg="scale=1 应该正好等于条件预测")
    eq(cfg_combine(u, c, 0.0), u, msg="scale=0 应该退化成无条件")
    eq(cfg_combine(u, c, 2.0), 2 * c - u)
    # 方向检查: scale 越大越远离无条件
    d1 = (cfg_combine(u, c, 3.0) - u).norm()
    d2 = (cfg_combine(u, c, 1.0) - u).norm()
    true(d1 > d2, "scale 变大应该离无条件预测更远，你的公式方向反了")


# ==================================================================
# 06  CFG 的 batch 技巧：一次前向算两个分支
# ==================================================================
def cfg_forward(model, x, t, cond, null_cond, scale):
    """不要跑两次 model。把 x 复制两份、条件拼成 [null, cond]，一次前向再拆开。
    x: (B,C,H,W)   cond: (B,D)   null_cond: (D,)
    model(x, t, c) -> 和 x 同形状
    返回组合后的 (B,C,H,W)。
    """
    # --- TODO 06 ---
    raise todo()


class _FakeEps(nn.Module):
    """给定的假模型: 输出 = x * 0 + 条件向量的均值，方便验证分支有没有拼对。"""
    def forward(self, x, t, c):
        return torch.zeros_like(x) + c.mean(-1).reshape(-1, 1, 1, 1)


@task(6, "CFG 的 batch 技巧")
def t06():
    m = _FakeEps()
    x = torch.randn(3, 1, 2, 2)
    t = torch.zeros(3, dtype=torch.long)
    cond = torch.full((3, 4), 2.0)
    null = torch.zeros(4)
    out = cfg_forward(m, x, t, cond, null, scale=3.0)
    shape_is(out, x.shape)
    # uncond=0, cond=2 -> 0 + 3*(2-0) = 6
    eq(out, torch.full_like(x, 6.0),
       msg="拼接顺序错了。约定是前一半 null、后一半 cond")
    calls = []
    class Counting(nn.Module):
        def forward(self, x, t, c):
            calls.append(x.shape[0])
            return torch.zeros_like(x) + c.mean(-1).reshape(-1, 1, 1, 1)
    cfg_forward(Counting(), x, t, cond, null, 3.0)
    true(len(calls) == 1 and calls[0] == 6,
         f"应该只调用一次 model、batch 翻倍到 6，实际调用 {len(calls)} 次 batch={calls}")


# ==================================================================
# 07  DDIM 的时间步子集
# ==================================================================
def ddim_timesteps(T, steps):
    """从 [0, T) 里等间隔取 steps 个时间步，**从大到小**返回 1D LongTensor。
    最后一个必须是 0（否则最后一步落不到干净图上）。
    T=1000, steps=5 -> tensor([800, 600, 400, 200, 0])
    """
    # --- TODO 07 ---
    raise todo()


@task(7, "DDIM 时间步子集")
def t07():
    ts = ddim_timesteps(1000, 5)
    true(ts.dtype == torch.long)
    eq(ts, torch.tensor([800, 600, 400, 200, 0]))
    ts2 = ddim_timesteps(1000, 50)
    true(len(ts2) == 50 and ts2[0] == 980 and ts2[-1] == 0)
    true((ts2[1:] < ts2[:-1]).all(), "必须严格递减")


# ==================================================================
# 08  DDIM 一步 —— 喂进真噪声应该精确还原
# ==================================================================
def ddim_step(x_t, t, t_prev, eps_pred, sched):
    """x0_pred = (x_t - sqrt(1-ab_t) * eps) / sqrt(ab_t)
    x_prev  = sqrt(ab_prev) * x0_pred + sqrt(1-ab_prev) * eps      (eta=0，不加随机噪声)
    t / t_prev 都是 (B,) 张量。t_prev 允许是 -1，此时 ab_prev = 1（直接返回 x0_pred）。
    """
    # --- TODO 08 ---
    raise todo()


@task(8, "DDIM 一步（精确还原检验）")
def t08():
    seed(2)
    s = linear_schedule(1000)
    x0 = torch.randn(4, 1, 8, 8)
    eps = torch.randn_like(x0)
    t = torch.full((4,), 600, dtype=torch.long)
    tp = torch.full((4,), 400, dtype=torch.long)
    xt = q_sample(x0, t, eps, s)
    got = ddim_step(xt, t, tp, eps, s)
    want = q_sample(x0, tp, eps, s)
    eq(got, want, tol=1e-4,
       msg="喂进当初真正用的那个噪声，DDIM 一步应该**精确**落在 q_sample(x0, t_prev) 上。"
           "这就是 DDIM 确定性的来源")
    # 走到底
    last = ddim_step(xt, t, torch.full((4,), -1, dtype=torch.long), eps, s)
    eq(last, x0, tol=1e-3, msg="t_prev=-1 时应该直接给出 x0")


# ==================================================================
# 09  完整 DDIM 采样循环
# ==================================================================
@torch.no_grad()
def ddim_sample(model, shape, sched, steps=10, device="cpu"):
    """从 N(0,I) 出发，按 ddim_timesteps 依次调 ddim_step。
    model(x, t) -> 预测噪声。返回最终的 x0，形状 = shape。
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "DDIM 采样循环")
def t09():
    seed(3)
    s = linear_schedule(100)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(1, 1, 3, 1, 1)
        def forward(self, x, t):
            return self.c(x) * 0.1

    m = Tiny().eval()
    torch.manual_seed(7)
    a = ddim_sample(m, (2, 1, 8, 8), s, steps=10)
    shape_is(a, (2, 1, 8, 8))
    torch.manual_seed(7)
    b = ddim_sample(m, (2, 1, 8, 8), s, steps=10)
    eq(a, b, msg="同一个起始噪声，DDIM 必须给出完全相同的结果（它没有随机性）")
    true(torch.isfinite(a).all(), "出现了 nan/inf")


# ==================================================================
# 10  guidance scale 的效果（数值上的可验证部分）
# ==================================================================
@task(10, "guidance 的单调性")
def t10():
    u = torch.zeros(1, 4)
    c = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    prev = None
    for sc in [1.0, 2.0, 5.0, 10.0]:
        d = (cfg_combine(u, c, sc) - u).norm().item()
        if prev is not None:
            true(d > prev, f"scale={sc} 时反而更靠近无条件了")
        prev = d


# ==================================================================
# 11  latent diffusion：把 VAE 接进来的形状账
# ==================================================================
LDM_SHAPES = {
    "图像 x":                    None,   # 512x512 RGB，batch 4
    "VAE 编码后的 latent":        None,   # 8 倍下采样，4 通道
    "latent 转成 token 序列":     None,   # 给 cross-attn 用，通道数 320
    "CLIP 文本嵌入":              None,   # 77 个 token，768 维
    "cross-attn 输出":            None,   # 和 latent token 同形
    "UNet 输出（预测的噪声）":     None,
}


@task(11, "latent diffusion 的形状账")
def t11():
    want = {
        "图像 x": (4, 3, 512, 512),
        "VAE 编码后的 latent": (4, 4, 64, 64),
        "latent 转成 token 序列": (4, 64 * 64, 320),
        "CLIP 文本嵌入": (4, 77, 768),
        "cross-attn 输出": (4, 64 * 64, 320),
        "UNet 输出（预测的噪声）": (4, 4, 64, 64),
    }
    wrong = [f"{k!r}: 你填 {v}, 应为 {want[k]}" for k, v in LDM_SHAPES.items()
             if v is None or tuple(v) != want[k]]
    true(not wrong, "\n       " + "\n       ".join(wrong))


# ==================================================================
# 12  找 bug —— 这个 CFG 采样有 3 处错
# ==================================================================
def cfg_sample_buggy(model, x, t, cond, null_cond, scale):
    both_x = torch.cat([x, x])
    both_c = torch.cat([cond, null_cond.expand(cond.size(0), -1)])
    both_t = torch.cat([t, t])
    out = model(both_x, both_t, both_c)
    e_a, e_b = out.chunk(2)
    return e_a + scale * (e_a - e_b)


@task(12, "找 bug: cfg_sample_buggy")
def t12():
    m = _FakeEps()
    x = torch.randn(3, 1, 2, 2)
    t = torch.zeros(3, dtype=torch.long)
    cond, null = torch.full((3, 4), 2.0), torch.zeros(4)
    for sc in [1.0, 3.0]:
        eq(cfg_sample_buggy(m, x, t, cond, null, sc),
           cfg_forward(m, x, t, cond, null, sc), tol=1e-5,
           msg="三处: 拼接顺序应为 [null, cond]；差值方向应为 (cond - uncond)；"
               "基准项应该是 uncond 不是 cond")


if __name__ == "__main__":
    raise SystemExit(run("Day 09 · 条件化与 CFG"))
