"""Day 05 · 从 GPT-2 升级成 Llama —— 13 题

运行:  python drills/day05.py
第 4 题和第 10 题是本天的核心，它们检验的是"性质"而不是"数值"。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, params

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 01  RMSNorm 模块
# ==================================================================
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight
    只有一个可学习参数 weight，形状 (dim,)，初始全 1。没有 bias，也不减均值。
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        # --- TODO 01 ---
        raise todo()


@task(1, "RMSNorm")
def t01():
    seed(0)
    m = RMSNorm(8)
    true(params(m) == 8, f"参数量应为 8（LayerNorm 是 16，这就是省下的一半）: {params(m)}")
    x = torch.randn(2, 5, 8) * 7 + 3
    y = m(x)
    eq(y.pow(2).mean(-1), torch.ones(2, 5), tol=1e-3, msg="输出的均方值应该 ≈ 1")
    # 和 LayerNorm 的关键区别: 均值不会被减掉
    true(abs(y.mean().item()) > 1e-3, "RMSNorm 不减均值，输出均值不该是 0")


# ==================================================================
# 02  RoPE 的频率表
# ==================================================================
def precompute_freqs(head_dim, max_seq_len, theta=10000.0):
    """返回 (cos, sin)，形状都是 (max_seq_len, head_dim // 2)。
    第 i 对维度的角速度 = 1 / theta ** (2i / head_dim)，i = 0, 1, ..., head_dim/2 - 1
    位置 t 的角度 = t * 角速度
    """
    # --- TODO 02 ---
    raise todo()


@task(2, "RoPE 频率表")
def t02():
    cos, sin = precompute_freqs(8, 16)
    shape_is(cos, (16, 4))
    shape_is(sin, (16, 4))
    eq(cos[0], torch.ones(4), msg="位置 0 的角度全是 0，cos 应该全 1")
    eq(sin[0], torch.zeros(4))
    # 第 0 对维度角速度为 1，所以位置 t 的角度就是 t
    eq(cos[3, 0], torch.tensor(math.cos(3.0)))
    eq(sin[5, 0], torch.tensor(math.sin(5.0)))
    # 最后一对维度转得最慢
    true(abs(sin[1, -1].item()) < abs(sin[1, 0].item()), "高维应该转得更慢")


# ==================================================================
# 03  应用 RoPE（实数版，相邻两维配对）
# ==================================================================
def apply_rope(x, cos, sin):
    """x: (B, nh, T, hd)   cos/sin: (T, hd//2)  ->  (B, nh, T, hd)
    配对方式是**相邻两维**: (x0,x1) 一对, (x2,x3) 一对, ...
        out[..., 2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        out[..., 2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
    """
    # --- TODO 03 ---
    raise todo()


@task(3, "应用 RoPE")
def t03():
    seed(1)
    x = torch.randn(2, 3, 6, 8)
    cos, sin = precompute_freqs(8, 6)
    y = apply_rope(x, cos, sin)
    shape_is(y, x.shape)
    eq(y[:, :, 0], x[:, :, 0], msg="位置 0 不该被旋转")
    # 旋转是保范的
    eq(y.norm(dim=-1), x.norm(dim=-1), tol=1e-4, msg="旋转必须保持向量长度不变")


# ==================================================================
# 04  RoPE 的核心性质：内积只依赖相对位置
# ==================================================================
@task(4, "RoPE 相对位置性质")
def t04():
    seed(2)
    cos, sin = precompute_freqs(16, 64)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def dot_at(m, n):
        qm = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum()

    eq(dot_at(3, 5), dot_at(20, 22), tol=1e-4,
       msg="q 在 3、k 在 5 的内积，应该等于 q 在 20、k 在 22 的内积（相对距离都是 2）。"
           "不相等说明 RoPE 写错了")
    true(not torch.allclose(dot_at(3, 5), dot_at(3, 9), atol=1e-4), "不同相对距离应该给出不同内积")


# ==================================================================
# 05  复数版 RoPE，和实数版对拍
# ==================================================================
def apply_rope_complex(x, freqs):
    """x: (B, nh, T, hd)   freqs: (T, hd//2) 的**角度**（不是 cos/sin）
    用 torch.view_as_complex / torch.polar / torch.view_as_real 实现，结果必须和 apply_rope 一致。
    """
    # --- TODO 05 ---
    raise todo()


@task(5, "复数版 RoPE 对拍实数版")
def t05():
    seed(3)
    x = torch.randn(2, 2, 5, 8)
    hd, T = 8, 5
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
    ang = torch.outer(torch.arange(T).float(), inv)      # (T, hd//2)
    eq(apply_rope_complex(x, ang), apply_rope(x, ang.cos(), ang.sin()), tol=1e-5)


# ==================================================================
# 06  SwiGLU
# ==================================================================
def swiglu_hidden(n_embd, multiple_of=256):
    """Llama 的隐层宽度: 先取 4*C 的 2/3（= 8C/3），再向上取整到 multiple_of 的倍数。"""
    # --- TODO 06 ---
    raise todo()


class SwiGLU(nn.Module):
    """三个无 bias 的 Linear:
        w1: C -> H     w3: C -> H     w2: H -> C
    forward: w2(silu(w1(x)) * w3(x))
    """

    def __init__(self, n_embd, hidden):
        super().__init__()
        # --- TODO 06b ---
        raise todo()


@task(6, "SwiGLU")
def t06():
    true(swiglu_hidden(4096, 256) == 11008, f"Llama-7B 的 FFN 宽度是 11008, 你算出 {swiglu_hidden(4096, 256)}")
    true(swiglu_hidden(64, 8) == 176, f"{swiglu_hidden(64, 8)}")
    m = SwiGLU(16, 32)
    shape_is(m(torch.randn(2, 5, 16)), (2, 5, 16))
    true(params(m) == 16 * 32 * 3, f"三个无 bias 的 Linear: {params(m)}")


# ==================================================================
# 07  GQA 的 repeat_kv
# ==================================================================
def repeat_kv(x, n_rep):
    """x: (B, n_kv_head, T, hd)  ->  (B, n_kv_head*n_rep, T, hd)
    每个 kv head 要连续重复 n_rep 次（head 0,0,1,1 而不是 0,1,0,1）。
    不许用 torch.cat 循环拼（用 expand + reshape）。
    """
    # --- TODO 07 ---
    raise todo()


@task(7, "GQA repeat_kv")
def t07():
    x = torch.arange(2 * 2 * 3 * 4, dtype=torch.float).reshape(2, 2, 3, 4)
    y = repeat_kv(x, 3)
    shape_is(y, (2, 6, 3, 4))
    eq(y[:, 0], x[:, 0])
    eq(y[:, 1], x[:, 0], msg="重复要连续: head 顺序应为 0,0,0,1,1,1")
    eq(y[:, 3], x[:, 1])
    eq(repeat_kv(x, 1), x)


# ==================================================================
# 08  KV-Cache
# ==================================================================
class KVCache:
    """用法:
        c = KVCache()
        k_all, v_all = c.update(k_new, v_new)   # 都是 (B, nh, T_new, hd)
    每次把新的 k/v 沿时间维拼到已有的后面，返回拼接后的完整 k/v。
    c.length 返回当前缓存的时间步数。c.reset() 清空。
    """

    def __init__(self):
        # --- TODO 08 ---
        raise todo()


@task(8, "KVCache")
def t08():
    c = KVCache()
    true(c.length == 0)
    k1, v1 = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    K, V = c.update(k1, v1)
    shape_is(K, (1, 2, 3, 4))
    true(c.length == 3)
    k2, v2 = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    K, V = c.update(k2, v2)
    shape_is(K, (1, 2, 4, 4))
    eq(K[:, :, :3], k1, msg="旧的缓存被覆盖了")
    eq(K[:, :, 3:], k2)
    true(c.length == 4)
    c.reset()
    true(c.length == 0)


# ==================================================================
# 09  带 cache 的单步 attention
# ==================================================================
def attn_step(q, k_all, v_all):
    """q: (B, nh, 1, hd) 只有当前这一步；k_all/v_all: (B, nh, T, hd) 全部历史。
    因为 q 只有一个位置而且它是最新的，**不需要 causal mask**（想清楚为什么）。
    返回 (B, nh, 1, hd)。
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "单步 attention 不需要 mask")
def t09():
    seed(4)
    q = torch.randn(1, 2, 1, 4)
    k, v = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
    eq(attn_step(q, k, v), F.scaled_dot_product_attention(q, k, v, is_causal=False))


# ==================================================================
# 10  KV-Cache 与全量前向必须逐 token 一致
# ==================================================================
def _full_attention(x, wq, wk, wv, cos, sin):
    """给定的参考实现：一次算完整个序列（带 RoPE 和 causal mask）。"""
    B, T, C = x.shape
    nh, hd = 2, C // 2
    q = (x @ wq).view(B, T, nh, hd).transpose(1, 2)
    k = (x @ wk).view(B, T, nh, hd).transpose(1, 2)
    v = (x @ wv).view(B, T, nh, hd).transpose(1, 2)
    q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
    return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)


def incremental_attention(x, wq, wk, wv, cos, sin):
    """一次喂一个 token，用 KVCache + attn_step 复现 _full_attention 的输出。
    返回 (B, T, C)。注意第 t 步的 RoPE 要用 cos[t:t+1] / sin[t:t+1]，不是 cos[0:1]。
    """
    # --- TODO 10 ---
    raise todo()


@task(10, "KV-Cache 输出必须与全量前向一致")
def t10():
    seed(5)
    B, T, C = 1, 7, 8
    x = torch.randn(B, T, C)
    wq, wk, wv = (torch.randn(C, C) * 0.3 for _ in range(3))
    cos, sin = precompute_freqs(C // 2, 32)
    eq(incremental_attention(x, wq, wk, wv, cos, sin),
       _full_attention(x, wq, wk, wv, cos, sin), tol=1e-4,
       msg="增量生成和全量前向对不上。九成是位置索引错了（每步都用了位置 0）")


# ==================================================================
# 11  采样：temperature / top-k / top-p
# ==================================================================
def apply_temperature(logits, t):
    """t 越小越确定。t 可能是 0，此时退化成 argmax（把最大值以外全设 -inf）。"""
    # --- TODO 11a ---
    raise todo()


def top_k_filter(logits, k):
    """保留最大的 k 个，其余置 -inf。logits: (..., V)"""
    # --- TODO 11b ---
    raise todo()


def top_p_filter(logits, p):
    """核采样。按概率从大到小排，累计概率**首次 >= p** 的那个 token 也保留，之后全部 -inf。
    至少保留 1 个。
    """
    # --- TODO 11c ---
    raise todo()


@task(11, "temperature / top-k / top-p")
def t11():
    lg = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    eq(apply_temperature(lg, 2.0), lg / 2)
    z = apply_temperature(lg, 0.0)
    true(torch.isinf(z[0, :3]).all() and z[0, 3] == 4.0, "t=0 应该退化成 argmax")

    f = top_k_filter(lg, 2)
    true(torch.isinf(f[0, :2]).all(), "只该留最大的 2 个")
    eq(f[0, 2:], lg[0, 2:])

    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    lgp = probs.log()
    g = top_p_filter(lgp, 0.7)
    kept = (~torch.isinf(g)).sum().item()
    true(kept == 2, f"0.5 + 0.3 = 0.8 首次超过 0.7，应该保留 2 个，你保留了 {kept} 个")
    true(top_p_filter(lgp, 0.01).isinf().sum().item() == 3, "p 很小也至少要留 1 个")


# ==================================================================
# 12  重复惩罚
# ==================================================================
def repetition_penalty(logits, prev_ids, penalty=1.2):
    """logits: (B, V)   prev_ids: (B, T) 已生成的 token。
    对出现过的 token: 正 logit 除以 penalty，负 logit 乘以 penalty（都变得更不可能）。
    不许用 python 循环遍历 batch。
    """
    # --- TODO 12 ---
    raise todo()


@task(12, "重复惩罚")
def t12():
    lg = torch.tensor([[2.0, -2.0, 1.0]])
    prev = torch.tensor([[0, 1]])
    out = repetition_penalty(lg, prev, 2.0)
    eq(out, torch.tensor([[1.0, -4.0, 1.0]]),
       msg="正数要除、负数要乘；没出现过的 token 不动")


# ==================================================================
# 13  找 bug —— 它想写"交错配对"版，却写成了另一套约定
# ==================================================================
# 说明: RoPE 有两种主流配对约定 —— 相邻两维配对（本文件 / Meta llama 原版）
#       和前后对半配对（HuggingFace 的 rotate_half）。两种都对，但必须和权重布局一致，
#       混着用就会静默出错。下面这段要改成和上面 apply_rope 一样的交错约定。
# ==================================================================
def apply_rope_buggy(x, cos, sin):
    hd = x.size(-1)
    x1 = x[..., :hd // 2]
    x2 = x[..., hd // 2:]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.cat([o1, o2], dim=-1)


@task(13, "找 bug: apply_rope_buggy")
def t13():
    seed(6)
    x = torch.randn(1, 2, 5, 8)
    cos, sin = precompute_freqs(8, 5)
    eq(apply_rope_buggy(x, cos, sin), apply_rope(x, cos, sin), tol=1e-5,
       msg="两处: 取分量时要按相邻两维（0-1, 2-3...）而不是前后对半；"
           "拼回去也要交错还原，不能直接 cat")


if __name__ == "__main__":
    raise SystemExit(run("Day 05 · Llama 架构"))
