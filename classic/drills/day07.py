"""Day 07 · 潜空间与 VAE —— 11 题

运行:  python drills/day07.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, params, close

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 01  卷积输出尺寸 —— 脑算，别跑代码
# ==================================================================
def conv_out(H, kernel, stride, padding):
    """返回 Conv2d 后的空间尺寸: floor((H + 2p - k)/s) + 1"""
    # --- TODO 01 ---
    raise todo()


CONV_SHAPES = {
    # (H_in, k, s, p) -> H_out
    (32, 3, 1, 1): None,     # 最常见的"不改变尺寸"配置
    (32, 4, 2, 1): None,     # 最常见的"减半"配置
    (32, 3, 2, 1): None,
    (32, 1, 1, 0): None,     # 1x1 卷积
    (7, 3, 2, 0): None,
}


@task(1, "卷积输出尺寸")
def t01():
    wrong = []
    for (H, k, s, p), guess in CONV_SHAPES.items():
        real = conv_out(H, k, s, p)
        ref = nn.Conv2d(1, 1, k, s, p)(torch.zeros(1, 1, H, H)).shape[-1]
        true(real == ref, f"conv_out({H},{k},{s},{p}) 公式算错: {real} vs 实测 {ref}")
        if guess != real:
            wrong.append(f"({H},{k},{s},{p}): 你填 {guess}, 实际 {real}")
    true(not wrong, "\n       " + "\n       ".join(wrong))


# ==================================================================
# 02  重参数化 —— 三行，但少了它梯度就传不回去
# ==================================================================
def reparameterize(mu, logvar):
    """z = mu + std * eps，其中 std = exp(0.5 * logvar)，eps ~ N(0, I)。
    注意网络输出的是 logvar 不是 std（为什么？因为 std 必须为正，logvar 可以是任意实数）。
    """
    # --- TODO 02 ---
    raise todo()


@task(2, "重参数化")
def t02():
    seed(0)
    mu = torch.zeros(20000, 3)
    logvar = torch.full((20000, 3), math.log(4.0))    # var=4 -> std=2
    z = reparameterize(mu, logvar)
    shape_is(z, mu.shape)
    true(abs(z.mean().item()) < 0.05, f"均值应该 ≈ 0, 实际 {z.mean().item():.3f}")
    true(abs(z.std().item() - 2.0) < 0.05,
         f"std 应该 ≈ 2（= exp(0.5*log4)）, 实际 {z.std().item():.3f}。"
         "算成 exp(logvar) 的话会得到 4")
    # 梯度必须能传回 mu 和 logvar
    mu2 = torch.zeros(4, 3, requires_grad=True)
    lv2 = torch.zeros(4, 3, requires_grad=True)
    reparameterize(mu2, lv2).sum().backward()
    true(mu2.grad is not None and lv2.grad is not None, "梯度断了 —— 你是不是直接 torch.normal 采样了？")
    true(lv2.grad.abs().sum() > 0, "logvar 拿不到梯度")


# ==================================================================
# 03  KL 闭式解
# ==================================================================
def kl_normal(mu, logvar):
    """q = N(mu, exp(logvar)) 相对 p = N(0, I) 的 KL，对每个样本的所有维度求和。
    返回 (B,)。闭式: 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
    """
    # --- TODO 03 ---
    raise todo()


@task(3, "KL 闭式解")
def t03():
    mu, lv = torch.zeros(2, 5), torch.zeros(2, 5)
    eq(kl_normal(mu, lv), torch.zeros(2), msg="q 就是 p 时 KL 必须是 0")
    k = kl_normal(torch.ones(1, 1), torch.zeros(1, 1))
    eq(k, torch.tensor([0.5]), msg="mu=1, var=1 时 KL = 0.5")
    true((kl_normal(torch.randn(8, 4) * 3, torch.randn(8, 4)) >= 0).all(), "KL 不可能是负的")


# ==================================================================
# 04  闭式解和蒙特卡洛估计必须一致
# ==================================================================
@task(4, "KL 闭式 vs 蒙特卡洛")
def t04():
    seed(1)
    mu = torch.tensor([[0.5, -1.0]])
    lv = torch.tensor([[0.3, -0.7]])
    closed = kl_normal(mu, lv)[0]

    n = 400000
    std = (0.5 * lv).exp()
    z = mu + std * torch.randn(n, 2)
    log_q = (-0.5 * ((z - mu) / std) ** 2 - lv * 0.5 - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_p = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1)
    mc = (log_q - log_p).mean()
    true(abs(closed.item() - mc.item()) < 0.02,
         f"闭式 {closed.item():.4f} vs 采样估计 {mc.item():.4f} 差太多，闭式写错了")


# ==================================================================
# 05  Encoder：把 32x32 压到 8x8
# ==================================================================
class Encoder(nn.Module):
    """结构（每层后面接 SiLU，最后一层不接）:
        Conv(3 -> 32, k4 s2 p1)   32x32 -> 16x16
        Conv(32 -> 64, k4 s2 p1)  16x16 -> 8x8
        Conv(64 -> 2*z_ch, k3 s1 p1)   输出通道拆成 mu 和 logvar 各 z_ch
    forward(x) 返回 (mu, logvar)，形状都是 (B, z_ch, 8, 8)
    """

    def __init__(self, z_ch=4):
        super().__init__()
        # --- TODO 05 ---
        raise todo()


@task(5, "Encoder")
def t05():
    m = Encoder(4)
    mu, lv = m(torch.randn(2, 3, 32, 32))
    shape_is(mu, (2, 4, 8, 8), "mu")
    shape_is(lv, (2, 4, 8, 8), "logvar")
    true(not torch.equal(mu, lv), "mu 和 logvar 应该是最后一层输出的两半，不是同一个东西")


# ==================================================================
# 06  Decoder：把 8x8 还原回 32x32
# ==================================================================
class Decoder(nn.Module):
    """结构（每层后接 SiLU，最后一层接 tanh）:
        Conv(z_ch -> 64, k3 s1 p1)
        Upsample(x2) + Conv(64 -> 32, k3 s1 p1)     8x8 -> 16x16
        Upsample(x2) + Conv(32 -> 3,  k3 s1 p1)     16x16 -> 32x32
    用 Upsample+Conv 而不是 ConvTranspose2d，能少很多棋盘格伪影。
    """

    def __init__(self, z_ch=4):
        super().__init__()
        # --- TODO 06 ---
        raise todo()


@task(6, "Decoder")
def t06():
    m = Decoder(4)
    y = m(torch.randn(2, 4, 8, 8))
    shape_is(y, (2, 3, 32, 32))
    true(y.min() >= -1.0 and y.max() <= 1.0, "最后要过 tanh，输出范围 [-1, 1]")


# ==================================================================
# 07  完整 VAE 的 loss
# ==================================================================
def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    """返回 (total, recon, kl) 三个标量。
      recon: 对每个样本把所有像素的平方误差**求和**，再对 batch 取平均
      kl   : 每个样本的 KL 求和，再对 batch 取平均
      total = recon + beta * kl
    （注意 recon 用 sum 不是 mean，否则和 KL 的量级差几千倍，beta 根本没法调）
    """
    # --- TODO 07 ---
    raise todo()


@task(7, "VAE loss")
def t07():
    x = torch.zeros(4, 3, 8, 8)
    xh = torch.full_like(x, 0.5)
    mu, lv = torch.zeros(4, 2, 2, 2), torch.zeros(4, 2, 2, 2)
    tot, rec, kl = vae_loss(x, xh, mu, lv, beta=1.0)
    eq(rec, torch.tensor(3 * 8 * 8 * 0.25), msg="recon 应该是每样本像素平方误差之和")
    eq(kl, torch.tensor(0.0))
    eq(tot, rec)
    tot2, _, _ = vae_loss(x, xh, torch.ones(4, 2, 2, 2), lv, beta=10.0)
    true(tot2 > tot, "beta 变大、mu 非零，总 loss 应该变大")


# ==================================================================
# 08  压缩比 —— 扩散模型能跑起来的全部原因
# ==================================================================
def compression(H, W, C_img, h, w, C_z):
    """返回 (像素元素数 / 潜变量元素数)。"""
    # --- TODO 08 ---
    raise todo()


@task(8, "潜空间压缩比")
def t08():
    true(close(compression(32, 32, 3, 8, 8, 4), 12.0), "32x32x3 -> 8x8x4 是 12 倍")
    true(close(compression(512, 512, 3, 64, 64, 4), 48.0),
         "SD 的 512x512 -> 64x64x4 是 48 倍。没有这一步，扩散在像素上根本训不动")


# ==================================================================
# 09  scaling_factor 是干什么的
# ==================================================================
def fit_scaling_factor(latents):
    """SD 里的 0.18215 是这么来的: 让 latent 乘上它之后**整体标准差 ≈ 1**。
    给一批 latent，返回这个系数。
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "scaling_factor")
def t09():
    seed(2)
    lat = torch.randn(64, 4, 8, 8) * 5.49      # 真实 SD VAE 的 latent std 大约就是这个量级
    s = fit_scaling_factor(lat)
    true(close(s, 0.182, 0.01), f"你算出 {s:.5f}，应该 ≈ 0.182（这就是 0.18215 的来历）")
    true(abs((lat * s).std().item() - 1.0) < 0.01, "乘完之后 std 应该 ≈ 1")


# ==================================================================
# 10  后验坍塌的判据
# ==================================================================
def collapsed_dims(mu, logvar, thresh=0.01):
    """返回"坍塌"的维度数: 这些维度上 KL 的平均值 < thresh，
    说明 encoder 对它们直接输出了标准正态，等于没编码任何信息。
    mu / logvar: (B, D)。按维度算平均 KL。
    """
    # --- TODO 10 ---
    raise todo()


@task(10, "后验坍塌检测")
def t10():
    B = 100
    mu = torch.zeros(B, 4)
    lv = torch.zeros(B, 4)
    mu[:, 0] = torch.randn(B) * 2      # 只有第 0 维在用
    true(collapsed_dims(mu, lv) == 3, f"应该检出 3 个坍塌维, 你检出 {collapsed_dims(mu, lv)}")
    true(collapsed_dims(torch.randn(B, 4) * 2, torch.zeros(B, 4)) == 0)


# ==================================================================
# 11  找 bug —— 这个 VAE 训练步有 3 处错
# ==================================================================
def vae_step_buggy(x, x_hat, mu, logvar, beta=1.0):
    std = torch.exp(logvar)
    z = mu + std * torch.randn_like(std)
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl = 0.5 * (mu ** 2 + torch.exp(logvar) - 1 - logvar).sum()
    return recon - beta * kl, z


@task(11, "找 bug: vae_step_buggy")
def t11():
    seed(3)
    B = 8
    x, x_hat = torch.randn(B, 3, 8, 8), torch.randn(B, 3, 8, 8)
    mu, lv = torch.randn(B, 4) * 0.5, torch.randn(B, 4) * 0.5
    loss, _ = vae_step_buggy(x, x_hat, mu, lv, beta=2.0)
    tot, _, _ = vae_loss(x, x_hat, mu, lv, beta=2.0)
    eq(loss, tot, tol=1e-3,
       msg="两处: KL 忘了对 batch 取平均 / recon 和 KL 应该相加而不是相减")
    n = 50000
    z0 = torch.zeros(n, 1)
    _, z = vae_step_buggy(z0, z0, torch.zeros(n, 2), torch.full((n, 2), math.log(9.0)))
    true(abs(z.std().item() - 3.0) < 0.1,
         f"第三处: 采样出的 std = {z.std().item():.2f}，应该是 3。std = exp(0.5*logvar)")


if __name__ == "__main__":
    raise SystemExit(run("Day 07 · VAE 与潜空间"))
