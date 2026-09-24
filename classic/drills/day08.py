"""Day 08 · DDPM 从零 —— 14 题

运行:  python drills/day08.py
第 5 题（闭式加噪 == 逐步加噪）是整个扩散模型的地基，务必做通。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, params, close

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 01  线性 beta 调度
# ==================================================================
def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    """返回 dict，至少包含:
        betas          (T,)   在 [beta_start, beta_end] 上线性取 T 个点
        alphas         (T,)   = 1 - betas
        alphas_cumprod (T,)   = alphas 的累积乘积（论文里的 alpha_bar）
    """
    # --- TODO 01 ---
    raise todo()


@task(1, "线性 beta 调度")
def t01():
    s = linear_schedule(1000)
    shape_is(s["betas"], (1000,))
    eq(s["betas"][0], torch.tensor(1e-4))
    eq(s["betas"][-1], torch.tensor(0.02))
    eq(s["alphas"], 1 - s["betas"])
    eq(s["alphas_cumprod"][0], s["alphas"][0])
    eq(s["alphas_cumprod"][4], s["alphas"][:5].prod())
    ab = s["alphas_cumprod"]
    true((ab[1:] < ab[:-1]).all(), "alpha_bar 必须单调递减")
    true(ab[-1] < 0.01, f"最后一步 alpha_bar 应该接近 0（信号被噪声淹没），实际 {ab[-1]:.4f}")


# ==================================================================
# 02  cosine 调度 —— 改进版，低分辨率上明显更好
# ==================================================================
def cosine_schedule(T, s=0.008):
    """f(t) = cos((t/T + s) / (1 + s) * pi/2) ** 2,  t = 0..T
    alphas_cumprod[i] = f(i+1) / f(0)
    betas[i] = 1 - alphas_cumprod[i] / alphas_cumprod[i-1]，并 clip 到 [0, 0.999]
    返回和第 1 题同样结构的 dict。
    """
    # --- TODO 02 ---
    raise todo()


@task(2, "cosine 调度")
def t02():
    c = cosine_schedule(1000)
    ab = c["alphas_cumprod"]
    shape_is(ab, (1000,))
    true(ab[0] > 0.999, f"第一步几乎不加噪, alpha_bar[0]={ab[0]:.5f}")
    true(ab[-1] < 0.01)
    true((ab[1:] <= ab[:-1]).all(), "必须单调不增")
    true((c["betas"] >= 0).all() and (c["betas"] <= 0.999).all(), "betas 要 clip 到 [0, 0.999]")
    # cosine 的特点: 中段掉得比 linear 慢
    lin = linear_schedule(1000)["alphas_cumprod"]
    true(ab[500] > lin[500], "cosine 在中段应该保留更多信号，这就是它更好的原因")


# ==================================================================
# 03  extract —— 扩散代码里出现频率最高的小工具
# ==================================================================
def extract(arr, t, x_shape):
    """arr: (T,) 的系数表；t: (B,) int64 的时间步；x_shape: 目标张量形状如 (B,C,H,W)。
    返回 (B, 1, 1, 1) 这样能和 x 广播的张量。
    """
    # --- TODO 03 ---
    raise todo()


@task(3, "extract 广播工具")
def t03():
    arr = torch.arange(10).float()
    t = torch.tensor([3, 7])
    o = extract(arr, t, (2, 3, 4, 4))
    shape_is(o, (2, 1, 1, 1))
    eq(o.flatten(), torch.tensor([3.0, 7.0]))
    o2 = extract(arr, t, (2, 5))
    shape_is(o2, (2, 1), "二维输入时应该是 (B, 1)")
    true((o * torch.ones(2, 3, 4, 4)).shape == (2, 3, 4, 4), "要能和 x 广播")


# ==================================================================
# 04  q_sample：一步跳到任意时刻 t
# ==================================================================
def q_sample(x0, t, noise, sched):
    """x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    x0/noise: (B,C,H,W)；t: (B,)。
    """
    # --- TODO 04 ---
    raise todo()


@task(4, "q_sample 闭式")
def t04():
    seed(0)
    s = linear_schedule(1000)
    x0 = torch.randn(4, 3, 8, 8)
    n = torch.randn_like(x0)
    t0 = torch.zeros(4, dtype=torch.long)
    xt = q_sample(x0, t0, n, s)
    true((xt - x0).abs().mean() < 0.02, "t=0 时几乎还是原图")
    tT = torch.full((4,), 999, dtype=torch.long)
    xT = q_sample(x0, tT, n, s)
    true((xT - n).abs().mean() < 0.1, "t=T 时几乎就是纯噪声")
    # 不同样本可以有不同的 t
    tm = torch.tensor([0, 500, 999, 250])
    shape_is(q_sample(x0, tm, n, s), x0.shape)


# ==================================================================
# 05  闭式加噪 == 逐步加噪（本天的核心）
# ==================================================================
@task(5, "闭式 == 逐步加噪")
def t05():
    seed(1)
    T = 200
    s = linear_schedule(T, 1e-4, 0.05)
    N = 20000
    x0 = torch.full((N, 1), 2.0)                 # 固定信号，方便看统计量
    step = 60

    # 逐步: x_t = sqrt(1-beta_t) * x_{t-1} + sqrt(beta_t) * eps
    x = x0.clone()
    for i in range(step + 1):
        b = s["betas"][i]
        x = (1 - b).sqrt() * x + b.sqrt() * torch.randn_like(x)

    ab = s["alphas_cumprod"][step]
    want_mean = (ab.sqrt() * 2.0).item()
    want_std = (1 - ab).sqrt().item()
    true(abs(x.mean().item() - want_mean) < 0.02,
         f"逐步加噪的均值 {x.mean().item():.4f} 应该 ≈ sqrt(alpha_bar)*x0 = {want_mean:.4f}")
    true(abs(x.std().item() - want_std) < 0.02,
         f"逐步加噪的 std {x.std().item():.4f} 应该 ≈ sqrt(1-alpha_bar) = {want_std:.4f}")

    # 你的 q_sample 必须给出同样的统计量
    xt = q_sample(x0, torch.full((N,), step, dtype=torch.long), torch.randn(N, 1), s)
    true(abs(xt.mean().item() - want_mean) < 0.02, "q_sample 的均值和逐步加噪对不上")
    true(abs(xt.std().item() - want_std) < 0.02, "q_sample 的方差和逐步加噪对不上")


# ==================================================================
# 06  正弦时间嵌入
# ==================================================================
def timestep_embedding(t, dim, max_period=10000):
    """t: (B,) 整数时间步  ->  (B, dim)
    half = dim // 2
    freqs = exp(-log(max_period) * arange(half) / half)
    args  = t[:, None] * freqs[None]
    返回 cat([cos(args), sin(args)], dim=-1)   （先 cos 后 sin）

    注意: arange 要写 device=t.device。不写在 CPU 上测不出问题，
    一搬到 GPU 就报 "Expected all tensors to be on the same device"。
    """
    # --- TODO 06 ---
    raise todo()


@task(6, "正弦时间嵌入")
def t06():
    e = timestep_embedding(torch.tensor([0, 1, 100]), 64)
    shape_is(e, (3, 64))
    eq(e[0, :32], torch.ones(32), msg="t=0 时 cos 部分应该全 1")
    eq(e[0, 32:], torch.zeros(32), msg="t=0 时 sin 部分应该全 0")
    true(not torch.allclose(e[1], e[2]), "不同 t 必须给出不同嵌入")
    # 低频维度对 t 的变化不敏感
    true((e[1, 31] - e[2, 31]).abs() < (e[1, 0] - e[2, 0]).abs(), "高维应该是低频")


# ==================================================================
# 07  FiLM：把时间信息注入卷积特征
# ==================================================================
class FiLM(nn.Module):
    """把 (B, t_dim) 的时间嵌入变成每个通道的 scale 和 shift:
        emb -> SiLU -> Linear(t_dim, 2*ch) -> 拆成 scale, shift
        h = h * (1 + scale) + shift        （注意是 1+scale，让初始接近恒等）
    forward(h, emb): h 是 (B, ch, H, W)
    """

    def __init__(self, t_dim, ch):
        super().__init__()
        # --- TODO 07 ---
        raise todo()


@task(7, "FiLM 条件注入")
def t07():
    m = FiLM(16, 8)
    h = torch.randn(2, 8, 4, 4)
    emb = torch.randn(2, 16)
    shape_is(m(h, emb), (2, 8, 4, 4))
    with torch.no_grad():                    # Linear 权重清零 -> scale=shift=0 -> 恒等
        for p in m.parameters():
            p.zero_()
    eq(m(h, emb), h, msg="scale/shift 为 0 时应该是恒等映射（这就是要写 1+scale 的原因）")


# ==================================================================
# 08  ResBlock
# ==================================================================
class ResBlock(nn.Module):
    """结构:
        h = Conv3x3(GroupNorm(8, in) -> SiLU -> x)
        h = FiLM(h, emb)
        h = Conv3x3(GroupNorm(8, out) -> SiLU -> h)
        return h + skip(x)      # in != out 时 skip 是 1x1 卷积，否则恒等
    """

    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        # --- TODO 08 ---
        raise todo()


@task(8, "ResBlock")
def t08():
    m = ResBlock(8, 8, 16)
    shape_is(m(torch.randn(2, 8, 16, 16), torch.randn(2, 16)), (2, 8, 16, 16))
    m2 = ResBlock(8, 16, 16)
    shape_is(m2(torch.randn(2, 8, 16, 16), torch.randn(2, 16)), (2, 16, 16, 16))
    true(any(isinstance(mm, nn.Conv2d) and mm.kernel_size == (1, 1) for mm in m2.modules()),
         "通道数变了，skip 分支需要一个 1x1 卷积")


# ==================================================================
# 09  下采样 / 上采样 / skip 拼接
# ==================================================================
def unet_skip_concat(up_feat, skip_feat):
    """UNet 解码侧的标准操作: 沿通道维拼接。
    up_feat: (B, C1, H, W)   skip_feat: (B, C2, H, W)  ->  (B, C1+C2, H, W)
    如果空间尺寸差 1（奇数尺寸下采样后常见），先把 up_feat 用 F.pad 补齐。
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "skip 拼接")
def t09():
    a, b = torch.randn(2, 8, 16, 16), torch.randn(2, 4, 16, 16)
    shape_is(unet_skip_concat(a, b), (2, 12, 16, 16))
    # 尺寸不齐的情况
    a2 = torch.randn(2, 8, 7, 7)
    b2 = torch.randn(2, 4, 8, 8)
    shape_is(unet_skip_concat(a2, b2), (2, 12, 8, 8), "尺寸对不齐时要先 pad")


# ==================================================================
# 10  一个能跑的 mini UNet
# ==================================================================
class MiniUNet(nn.Module):
    """结构（32x32 输入）:
        t_mlp:  timestep_embedding(t, 64) -> Linear(64,128) -> SiLU -> Linear(128,128)
        stem:   Conv3x3(in_ch -> 32)
        down1:  ResBlock(32, 32)  存 skip  -> Conv stride2 到 16x16
        down2:  ResBlock(32, 64)  存 skip  -> Conv stride2 到 8x8
        mid:    ResBlock(64, 64)
        up2:    Upsample -> concat(skip2) -> ResBlock(64+64, 32)
        up1:    Upsample -> concat(skip1) -> ResBlock(32+32, 32)
        out:    GroupNorm -> SiLU -> Conv3x3(32 -> in_ch)
    forward(x, t) -> 和 x 同形状（预测的噪声）
    """

    def __init__(self, in_ch=1, t_dim=128):
        super().__init__()
        # --- TODO 10 ---
        raise todo()


@task(10, "MiniUNet 前向")
def t10():
    m = MiniUNet(1)
    x = torch.randn(2, 1, 32, 32)
    t = torch.randint(0, 1000, (2,))
    shape_is(m(x, t), (2, 1, 32, 32), "UNet 的输出必须和输入同形状（它预测的是噪声）")
    # 不同的 t 必须给出不同的输出，否则时间条件根本没接上
    o1 = m(x, torch.zeros(2, dtype=torch.long))
    o2 = m(x, torch.full((2,), 999, dtype=torch.long))
    true(not torch.allclose(o1, o2, atol=1e-4), "换了时间步输出没变 —— 时间嵌入没接进网络")


# ==================================================================
# 11  训练目标：就一行
# ==================================================================
def ddpm_loss(model, x0, sched, T):
    """1) 每个样本随机采一个 t ~ U[0, T)
       2) 采噪声 noise ~ N(0, I)
       3) x_t = q_sample(x0, t, noise)
       4) 返回 mse(model(x_t, t), noise)
    """
    # --- TODO 11 ---
    raise todo()


@task(11, "DDPM 训练目标")
def t11():
    seed(2)
    s = linear_schedule(100)
    m = MiniUNet(1)
    loss = ddpm_loss(m, torch.randn(4, 1, 32, 32), s, 100)
    true(loss.dim() == 0, "loss 要是标量")
    true(loss.item() > 0.1, "随机初始化的模型预测噪声，loss 应该在 1 附近")
    # 目标必须是 noise 而不是 x0
    class Zero(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)
    l0 = ddpm_loss(Zero(), torch.randn(64, 1, 8, 8) * 10, s, 100)
    true(abs(l0.item() - 1.0) < 0.15,
         f"永远输出 0 的模型，loss 应该 ≈ 噪声的方差 = 1，实际 {l0.item():.3f}。"
         "如果远大于 1，说明你的目标写成了 x0")


# ==================================================================
# 12  反向一步 p_sample
# ==================================================================
def p_sample_step(x_t, t, eps_pred, sched, noise=None):
    """DDPM 反向一步:
        mean = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_t)
        t > 0 时:  x_{t-1} = mean + sqrt(beta_t) * noise
        t == 0 时: x_0 = mean          （最后一步不加噪声）
    t 是 (B,) 张量；noise 为 None 时自己采。
    """
    # --- TODO 12 ---
    raise todo()


@task(12, "p_sample 一步")
def t12():
    seed(3)
    s = linear_schedule(100)
    x = torch.randn(4, 1, 4, 4)
    eps = torch.randn_like(x)
    n = torch.randn_like(x)
    t50 = torch.full((4,), 50, dtype=torch.long)
    a = p_sample_step(x, t50, eps, s, noise=n)
    shape_is(a, x.shape)
    b = p_sample_step(x, t50, eps, s, noise=torch.zeros_like(x))
    true(not torch.allclose(a, b), "t>0 时应该加噪声")
    t0 = torch.zeros(4, dtype=torch.long)
    c1 = p_sample_step(x, t0, eps, s, noise=n)
    c2 = p_sample_step(x, t0, eps, s, noise=torch.randn_like(x))
    eq(c1, c2, msg="最后一步（t=0）不能加噪声，否则生成的图会永远带一层沙")
    # 完美预测噪声时，一步应该把 x_t 拉回接近 x0 的方向
    x0 = torch.ones(1, 1, 2, 2)
    tt = torch.tensor([1])
    xt = q_sample(x0, tt, eps[:1, :, :2, :2], s)
    back = p_sample_step(xt, tt, eps[:1, :, :2, :2], s, noise=torch.zeros_like(xt))
    true((back - x0).abs().mean() < 0.2, "喂进真实噪声，反向一步应该接近 x0")


# ==================================================================
# 13  信噪比
# ==================================================================
def snr(sched, t):
    """SNR(t) = alpha_bar_t / (1 - alpha_bar_t)。t 是 (B,)，返回 (B,)。"""
    # --- TODO 13 ---
    raise todo()


@task(13, "信噪比")
def t13():
    s = linear_schedule(1000)
    v = snr(s, torch.tensor([0, 500, 999]))
    shape_is(v, (3,))
    true(v[0] > v[1] > v[2], "SNR 必须随 t 单调下降")
    true(v[0] > 1000, "t=0 时几乎没噪声，SNR 应该很大")
    true(v[-1] < 0.01, "t=T 时几乎全是噪声")


# ==================================================================
# 14  找 bug —— 这个加噪函数有 2 处错
# ==================================================================
def q_sample_buggy(x0, t, noise, sched):
    a = extract(sched["alphas"], t, x0.shape)
    return a * x0 + (1 - a) * noise


@task(14, "找 bug: q_sample_buggy")
def t14():
    seed(4)
    s = linear_schedule(1000)
    x0, n = torch.randn(4, 1, 8, 8), torch.randn(4, 1, 8, 8)
    for tv in [0, 300, 999]:
        t = torch.full((4,), tv, dtype=torch.long)
        eq(q_sample_buggy(x0, t, n, s), q_sample(x0, t, n, s), tol=1e-5,
           msg="两处: 系数表应该用 alphas_cumprod 而不是 alphas；两个系数都要开根号")


if __name__ == "__main__":
    raise SystemExit(run("Day 08 · DDPM 与 UNet"))
