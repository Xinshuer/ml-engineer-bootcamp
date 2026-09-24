"""Day 10 · DiT + Flow Matching：文生图其实就是你写过的 Transformer —— 12 题

运行:  python drills/day10.py
今天你会把 Day 4 的 Block 原样搬过来做图像生成。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, params, close

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 已给出：Day 4 的 attention，去掉 causal mask 就是图像用的版本
# ==================================================================
class SelfAttention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)      # 唯一的区别: is_causal=False
        return self.proj(o.transpose(1, 2).reshape(B, T, C))


# ==================================================================
# 01  patchify：把图片切成 token
# ==================================================================
def patchify(x, p):
    """x: (B, C, H, W)，H 和 W 都能被 p 整除。
    -> (B, N, p*p*C)，N = (H/p) * (W/p)，patch 按行优先排列。
    每个 patch 内部的展平顺序: 先 C，再 ph，再 pw（即 reshape 后 permute 到 (B, nh, nw, C, ph, pw) 再 flatten）。
    """
    # --- TODO 01 ---
    raise todo()


def unpatchify(tok, p, H, W, C):
    """patchify 的逆操作。"""
    # --- TODO 01b ---
    raise todo()


@task(1, "patchify / unpatchify 往返")
def t01():
    x = torch.randn(2, 3, 8, 8)
    tok = patchify(x, 2)
    shape_is(tok, (2, 16, 12), "8/2=4, 4*4=16 个 patch，每个 2*2*3=12 维")
    eq(unpatchify(tok, 2, 8, 8, 3), x, msg="往返必须完全还原")
    # 256x256 的图用 patch=16 -> 256 个 token，正好是 DiT 的常用配置
    shape_is(patchify(torch.randn(1, 4, 32, 32), 2), (1, 256, 16))


# ==================================================================
# 02  二维正弦位置编码
# ==================================================================
def pos_embed_2d(dim, h, w):
    """返回 (h*w, dim)。做法: 高和宽各用 dim/2 维的一维正弦编码，然后拼起来。
    一维部分用: freqs = 1 / 10000 ** (arange(0, d, 2) / d)，emb = cat([sin(pos*freqs), cos(pos*freqs)])
    token 顺序行优先。
    """
    # --- TODO 02 ---
    raise todo()


@task(2, "二维位置编码")
def t02():
    pe = pos_embed_2d(64, 4, 4)
    shape_is(pe, (16, 64))
    true(not torch.allclose(pe[0], pe[1]), "同一行相邻的两个 token 编码必须不同")
    true(not torch.allclose(pe[0], pe[4]), "同一列相邻的两个 token 编码必须不同")
    # 前一半只依赖行，后一半只依赖列
    eq(pe[0, :32], pe[3, :32], msg="前一半应该只编码行坐标，token 0 和 3 同在第 0 行")
    eq(pe[0, 32:], pe[12, 32:], msg="后一半应该只编码列坐标，token 0 和 12 同在第 0 列")


# ==================================================================
# 03  modulate：adaLN 的核心一行
# ==================================================================
def modulate(x, shift, scale):
    """x: (B, N, D)   shift/scale: (B, D)
    返回 x * (1 + scale) + shift。注意 1+ 这个偏移，它让 scale=0 时是恒等。
    """
    # --- TODO 03 ---
    raise todo()


@task(3, "modulate")
def t03():
    x = torch.randn(2, 5, 8)
    z = torch.zeros(2, 8)
    eq(modulate(x, z, z), x, msg="shift=scale=0 时必须是恒等")
    eq(modulate(x, torch.ones(2, 8), z), x + 1)
    eq(modulate(x, z, torch.ones(2, 8)), x * 2)


# ==================================================================
# 04  adaLN-Zero 的调制参数
# ==================================================================
class AdaLNModulation(nn.Module):
    """把条件向量 c (B, D) 变成 6 组调制参数:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    结构: SiLU -> Linear(D, 6D)，且这个 Linear 的**权重和 bias 都初始化为 0**
    （这就是 "adaLN-Zero" 的 Zero：训练一开始整个 block 是恒等映射）
    """

    def __init__(self, dim):
        super().__init__()
        # --- TODO 04 ---
        raise todo()


@task(4, "adaLN-Zero 调制参数")
def t04():
    m = AdaLNModulation(16)
    out = m(torch.randn(3, 16))
    true(isinstance(out, (tuple, list)) and len(out) == 6, "应该返回 6 个张量")
    for i, o in enumerate(out):
        shape_is(o, (3, 16), f"第 {i} 个")
    for o in out:
        eq(o, torch.zeros(3, 16), msg="零初始化：一开始 6 个参数全应该是 0")


# ==================================================================
# 05  DiTBlock —— 把 Day 4 的 Block 换个条件注入方式
# ==================================================================
class DiTBlock(nn.Module):
    """结构:
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = modulation(c)
        x = x + g_msa.unsqueeze(1) * attn(modulate(norm1(x), s_msa, sc_msa))
        x = x + g_mlp.unsqueeze(1) * mlp( modulate(norm2(x), s_mlp, sc_mlp))
    norm1/norm2 是 LayerNorm(dim, elementwise_affine=False)（缩放交给 adaLN 了）。
    mlp: Linear(dim, 4dim) -> GELU -> Linear(4dim, dim)
    对比 Day 4:  x = x + attn(ln1(x))  —— 多的只有 modulate 和 gate 两处。
    """

    def __init__(self, dim, n_head):
        super().__init__()
        # --- TODO 05 ---
        raise todo()


@task(5, "DiTBlock")
def t05():
    seed(0)
    b = DiTBlock(32, 4)
    x, c = torch.randn(2, 9, 32), torch.randn(2, 32)
    shape_is(b(x, c), (2, 9, 32))
    eq(b(x, c), x, tol=1e-5,
       msg="零初始化的 gate 让整个 block 一开始就是恒等映射。不相等说明 gate 没接上，"
           "或者 modulation 的 Linear 没有零初始化")
    # 训练起来之后（人为给 gate 一个非零值）就不再是恒等了
    with torch.no_grad():
        for p in b.modulation.parameters():
            p.normal_(0, 0.3)
    true(not torch.allclose(b(x, c), x, atol=1e-4), "gate 非零后应该真的改变 x")


# ==================================================================
# 06  Rectified Flow：训练数据的构造
# ==================================================================
def flow_interpolate(x0_noise, x1_data, t):
    """x_t = (1 - t) * noise + t * data
    x0_noise / x1_data: (B, C, H, W)   t: (B,) 取值 [0, 1] 的浮点。
    """
    # --- TODO 06 ---
    raise todo()


def flow_target(x0_noise, x1_data):
    """速度场的目标: v = data - noise（这是一条直线，从噪声直接指向数据）。"""
    # --- TODO 06b ---
    raise todo()


@task(6, "Rectified Flow 插值与目标")
def t06():
    n, d = torch.randn(4, 1, 4, 4), torch.randn(4, 1, 4, 4)
    eq(flow_interpolate(n, d, torch.zeros(4)), n, msg="t=0 应该是纯噪声")
    eq(flow_interpolate(n, d, torch.ones(4)), d, msg="t=1 应该是干净数据")
    eq(flow_interpolate(n, d, torch.full((4,), 0.5)), (n + d) / 2)
    eq(flow_target(n, d), d - n)
    # 关键性质: 沿 t 求导恒等于目标速度
    t1, t2 = torch.full((4,), 0.3), torch.full((4,), 0.3001)
    num = (flow_interpolate(n, d, t2) - flow_interpolate(n, d, t1)) / 0.0001
    eq(num, flow_target(n, d), tol=1e-2, msg="x_t 对 t 的导数应该正好是目标速度")


# ==================================================================
# 07  Flow 的训练目标：5 行
# ==================================================================
def flow_loss(model, x1_data):
    """1) t ~ U(0,1)，每个样本一个
       2) noise ~ N(0, I)
       3) x_t = 插值
       4) 返回 mse(model(x_t, t), data - noise)
    model(x, t) -> 和 x 同形状。
    """
    # --- TODO 07 ---
    raise todo()


@task(7, "Flow 训练目标")
def t07():
    seed(1)

    class Zero(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    # data 也是标准正态时，v = data - noise 的方差是 2
    l = flow_loss(Zero(), torch.randn(4096, 1, 4, 4))
    true(abs(l.item() - 2.0) < 0.15,
         f"永远输出 0 的模型，loss 应该 ≈ Var(data - noise) = 2，实际 {l.item():.3f}")
    true(l.dim() == 0)


# ==================================================================
# 08  欧拉采样
# ==================================================================
@torch.no_grad()
def flow_sample(model, shape, steps=20, device="cpu"):
    """从 t=0 的纯噪声出发，用欧拉法积分到 t=1:
        dt = 1 / steps
        for i in range(steps):  t = i * dt;  x = x + model(x, t_batch) * dt
    返回最终的 x。
    """
    # --- TODO 08 ---
    raise todo()


@task(8, "欧拉采样")
def t08():
    class ConstV(nn.Module):
        """速度场恒等于 5：从 x 出发积分 1 个单位时间，结果必须是 x + 5，
        而且和步数无关（这是检验积分写对没有的最好办法）。"""
        def forward(self, x, t):
            return torch.full_like(x, 5.0)

    m = ConstV()
    torch.manual_seed(0)
    a = flow_sample(m, (2, 1, 3, 3), steps=4)
    torch.manual_seed(0)
    b = flow_sample(m, (2, 1, 3, 3), steps=100)
    eq(a, b, tol=1e-4, msg="常速度场下，结果不该随步数变化 —— 你的 dt 算错了")
    torch.manual_seed(0)
    x0 = torch.randn(2, 1, 3, 3)
    eq(a, x0 + 5.0, tol=1e-4, msg="积分了 1 个单位时间，位移应该正好是速度 × 1")


# ==================================================================
# 09  Flow 和 DDPM 的代码量对比（自评）
# ==================================================================
# 数一下你写的行数（不含空行和注释）:
LOC = {
    "DDPM 的 q_sample + p_sample_step": None,
    "Flow 的 flow_interpolate + 欧拉一步": None,
}


@task(9, "代码量对比（自评）")
def t09():
    true(all(v is not None for v in LOC.values()), "还没填。数完再填，这一题是让你有个体感")
    true(LOC["Flow 的 flow_interpolate + 欧拉一步"] < LOC["DDPM 的 q_sample + p_sample_step"],
         "Flow 那边应该明显更短 —— 这就是它现在成为主流的一大原因")


# ==================================================================
# 10  完整的 mini DiT
# ==================================================================
class MiniDiT(nn.Module):
    """结构:
        x -> patchify(p) -> Linear(p*p*C, dim) -> + pos_embed_2d
        c = t_embed(t) + (可选的 class 嵌入)     这里只做 t
        n_layer 个 DiTBlock(x, c)
        LayerNorm(no affine) -> modulate(最后一组 shift/scale) -> Linear(dim, p*p*C) -> unpatchify
    forward(x, t) -> 和 x 同形状
    t_embed: 复用 Day 8 的 timestep_embedding(t, dim) 再过一个两层 MLP。
    最后那个 Linear(dim, p*p*C) 和输出侧的 modulate 层都要**零初始化**（DiT 论文的做法：
    训练开始时整个网络输出恒为 0，梯度从一个干净的起点长出来）。
    """

    def __init__(self, in_ch=4, img=8, patch=2, dim=64, n_layer=2, n_head=4):
        super().__init__()
        # --- TODO 10 ---
        raise todo()


@task(10, "MiniDiT 前向")
def t10():
    seed(2)
    m = MiniDiT(in_ch=4, img=8, patch=2, dim=64, n_layer=2)
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    shape_is(m(x, t), (2, 4, 8, 8))
    eq(m(x, t), torch.zeros(2, 4, 8, 8),
       msg="零初始化的输出层让训练前的输出恒为 0 —— 这是对的，不是 bug")
    # 给所有参数加一点扰动，模拟"训练了几步之后"，再检查时间条件有没有真的接进去
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    o1, o2 = m(x, torch.zeros(2)), m(x, torch.ones(2))
    true(not torch.allclose(o1, o2, atol=1e-5), "换了 t 输出没变 —— 时间条件没接上")


# ==================================================================
# 11  复用检验：DiT 里的 attention 和 Day 4 的是不是同一段代码
# ==================================================================
# 打开 drills/day04.py 里你写的 CausalSelfAttention，和本文件顶部给出的 SelfAttention 对比。
# 确认两段代码的差别只有: (a) is_causal / mask，(b) 没有 block_size buffer。
# 确认之后把下面改成 True。
REUSE_CONFIRMED = False


@task(11, "复用检验（自评）")
def t11():
    true(REUSE_CONFIRMED,
         "去对比一下。确认'文生图模型 = 你写过的 Transformer' 是今天唯一真正重要的事")


# ==================================================================
# 12  找 bug —— 这个 DiTBlock 有 4 处错
# ==================================================================
class DiTBlockBuggy(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_head)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.normal_(self.modulation[1].weight, std=0.02)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.modulation(c).chunk(6, dim=-1)
        x = x + self.attn(self.norm1(x) * sc1.unsqueeze(1) + s1.unsqueeze(1))
        x = x + g2.unsqueeze(1) * self.mlp(self.norm2(x) * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1))
        return x


@task(12, "找 bug: DiTBlockBuggy")
def t12():
    seed(3)
    b = DiTBlockBuggy(32, 4)
    x, c = torch.randn(2, 9, 32), torch.randn(2, 32)
    eq(b(x, c), x, tol=1e-5,
       msg="其中三处: modulation 的 Linear 要零初始化；attn 那一路漏了 gate；"
           "attn 那一路的 scale 要写成 (1 + scale)")
    ln = [m for m in b.modules() if isinstance(m, nn.LayerNorm)]
    true(all(m.elementwise_affine is False for m in ln),
         "第四处: adaLN 已经负责缩放了，两个 LayerNorm 都应该 elementwise_affine=False")


if __name__ == "__main__":
    raise SystemExit(run("Day 10 · DiT 与 Flow Matching"))
