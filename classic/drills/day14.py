"""Day 14 · 把任意一篇论文变成代码 —— 11 题

运行:  python drills/day14.py
今天没有新架构。今天练的是"拿到一个没见过的东西，怎么在几小时内跑通"。
第 5、6 题是没在前 13 天出现过的模块，故意的 —— 那才是这一天的考试。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, close, params

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 01  五问法 —— 读 paper 只找这五件事
# ==================================================================
# 下面是一篇你没读过的论文的核心描述，按五问法填表。
#
#   「我们把图像编码成 H×W 个特征向量，用一个大小为 K 的可学习码本对每个向量做最近邻
#    量化，用直通估计器回传梯度。训练目标是重建误差加上两个 stop-gradient 项：
#    ||sg[z_e] - e||² 和 β||z_e - sg[e]||²。生成时先训练一个自回归先验模型在离散
#    码上采样，再解码成图像。」
#
# 填 "add" / "cross-attn" / "adaln" / "concat" / "none" 之一表示条件注入方式；
# 填 "single" / "autoregressive" / "iterative-denoise" 之一表示采样方式。
FIVE_QUESTIONS = {
    "输入 shape":      None,   # 填 tuple，用 B/C/H/W 的具体数: batch 8, RGB, 64x64
    "输出 shape":      None,   # 重建出来的图
    "潜变量 shape":     None,   # 编码后 16x16 个离散码，batch 8（离散码没有通道维）
    "loss 有几项":      None,   # 填 int
    "条件注入方式":     None,
    "采样方式":         None,
}


@task(1, "五问法填表")
def t01():
    want = {
        "输入 shape": (8, 3, 64, 64),
        "输出 shape": (8, 3, 64, 64),
        "潜变量 shape": (8, 16, 16),
        "loss 有几项": 3,
        "条件注入方式": "none",
        "采样方式": "autoregressive",
    }
    wrong = []
    for k, v in FIVE_QUESTIONS.items():
        w = want[k]
        got = tuple(v) if isinstance(v, (list, tuple)) else v
        if v is None or got != w:
            wrong.append(f"{k!r}: 你填 {v}, 应为 {w}")
    true(not wrong, "\n       " + "\n       ".join(wrong))


# ==================================================================
# 02  smoke test 模板 —— 写完 model.py 的第一件事
# ==================================================================
def smoke_test(model, *input_shapes, expect_shape=None):
    """喂随机张量跑一次 forward，检查:
      1) 不报错
      2) 输出 shape == expect_shape（给了的话）
      3) 输出里没有 nan / inf
    返回 True，任何一条不满足就 raise AssertionError（信息要说清是哪条）。
    输入按 input_shapes 依次用 torch.randn 造。
    """
    # --- TODO 02 ---
    raise todo()


@task(2, "smoke test 工具")
def t02():
    ok = nn.Linear(4, 3)
    true(smoke_test(ok, (2, 4), expect_shape=(2, 3)) is True)

    class BadShape(nn.Module):
        def forward(self, x):
            return x
    try:
        smoke_test(BadShape(), (2, 4), expect_shape=(2, 3))
        true(False, "shape 不对时应该 raise")
    except AssertionError:
        pass

    class Nan(nn.Module):
        def forward(self, x):
            return x * float("nan")
    try:
        smoke_test(Nan(), (2, 4), expect_shape=(2, 4))
        true(False, "输出是 nan 时应该 raise")
    except AssertionError:
        pass


# ==================================================================
# 03  在一个 batch 上过拟合 —— 全世界最强的调试手段
# ==================================================================
def overfit_one_batch(model, x, y, steps=300, lr=1e-2):
    """反复在同一个 (x, y) 上训练。返回 loss 列表（float）。
    用 AdamW + F.cross_entropy。这一步跑不到接近 0，说明模型/loss 写错了，
    别去调数据、别去调超参 —— 先回来改代码。
    """
    # --- TODO 03 ---
    raise todo()


@task(3, "过拟合单个 batch")
def t03():
    seed(0)
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 5))
    x, y = torch.randn(16, 8), torch.randint(0, 5, (16,))
    losses = overfit_one_batch(model, x, y)
    true(isinstance(losses[0], float), "要存 .item()，不是 tensor")
    true(losses[-1] < 0.01,
         f"最终 loss = {losses[-1]:.4f}。一个 16 样本的 batch 应该能被轻松记住。"
         "记不住就是代码有问题")
    true(losses[-1] < losses[0], "loss 没下降")


# ==================================================================
# 04  梯度裁剪
# ==================================================================
def clip_and_report(model, max_norm):
    """裁剪梯度，返回**裁剪前**的总梯度范数（float）。
    用 torch.nn.utils.clip_grad_norm_，注意它的返回值正是裁剪前的范数。
    """
    # --- TODO 04 ---
    raise todo()


@task(4, "梯度裁剪")
def t04():
    m = nn.Linear(4, 2)
    m.weight.grad = torch.full_like(m.weight, 3.0)
    m.bias.grad = torch.zeros_like(m.bias)
    before = clip_and_report(m, 1.0)
    true(close(before, math.sqrt(8 * 9), 1e-4), f"裁剪前范数应为 sqrt(72)≈8.485, 你返回 {before}")
    after = m.weight.grad.norm().item()
    true(close(after, 1.0, 1e-4), f"裁剪后范数应为 1.0, 实际 {after:.4f}")


# ==================================================================
# 05  从公式到代码 (1)：VQ 的直通估计器
# ==================================================================
# 这个东西前 13 天一次都没讲过。只给公式，你来写。
#   量化:   e = codebook[argmin_k ||z - codebook[k]||]
#   直通:   z_q = z + (e - z).detach()
#           —— 前向等于 e，反向梯度直接绕过量化传回 z
def vq_quantize(z, codebook):
    """z: (B, D)   codebook: (K, D)
    返回 (z_q, indices)：z_q 是 (B, D)，indices 是 (B,) int64。
    z_q 必须用直通估计器，让梯度能传回 z。
    """
    # --- TODO 05 ---
    raise todo()


@task(5, "VQ 直通估计器（没讲过）")
def t05():
    cb = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    z = torch.tensor([[0.1, 0.1], [0.9, 1.2], [-0.8, 1.9]], requires_grad=True)
    zq, idx = vq_quantize(z, cb)
    eq(idx, torch.tensor([0, 1, 2]), msg="最近邻找错了")
    eq(zq, cb[idx], msg="前向输出必须精确等于码本向量")
    zq.sum().backward()
    eq(z.grad, torch.ones_like(z),
       msg="梯度必须原样穿过量化传回 z。如果 grad 是 0，说明你没写直通（少了 z + (e-z).detach()）")


# ==================================================================
# 06  从公式到代码 (2)：VQ 的三项 loss
# ==================================================================
def vq_loss(z, z_q_hard, beta=0.25):
    """z: encoder 输出 (B, D)（带梯度）   z_q_hard: 量化后的码本向量 (B, D)
    codebook loss  = ||sg[z] - e||²        （只更新码本）
    commitment loss= beta * ||z - sg[e]||²  （只更新 encoder）
    返回两项之和（都用 mse_loss 的默认 mean）。
    """
    # --- TODO 06 ---
    raise todo()


@task(6, "VQ 的 stop-gradient 两项（没讲过）")
def t06():
    z = torch.tensor([[1.0, 0.0]], requires_grad=True)
    e = torch.tensor([[0.0, 0.0]], requires_grad=True)
    # codebook 项 = mean((1-0)^2, (0-0)^2) = 0.5
    # commitment  = 0.25 * 0.5 = 0.125
    eq(vq_loss(z, e, beta=0.25), torch.tensor(0.625),
       msg="两项都用 mse_loss 默认的 mean，然后相加")
    eq(vq_loss(z, e, beta=0.0), torch.tensor(0.5), msg="beta=0 时只剩 codebook 项")


# ==================================================================
# 07  梯度流向：sg 到底挡住了谁
# ==================================================================
@task(7, "VQ loss 的梯度流向（没讲过）")
def t07():
    # codebook 项只给 e 梯度，commitment 项只给 z 梯度
    z = torch.tensor([[2.0, 0.0]], requires_grad=True)
    e = torch.tensor([[0.0, 0.0]], requires_grad=True)
    l = vq_loss(z, e, beta=0.25)
    l.backward()
    # ||sg[z]-e||^2 对 e 的梯度 = -2(z-e)/D = -2*2/2 = -2  （mse_loss 默认 mean，D=2）
    eq(e.grad, torch.tensor([[-2.0, 0.0]]),
       msg="codebook 项写错了：它应该是 ||sg[z] - e||²，只有 e 带梯度")
    # beta*||z-sg[e]||^2 对 z 的梯度 = beta*2(z-e)/D = 0.25*2*2/2 = 0.5
    eq(z.grad, torch.tensor([[0.5, 0.0]]),
       msg="commitment 项写错了：它应该是 beta*||z - sg[e]||²，只有 z 带梯度")


# ==================================================================
# 08  症状 -> 病因
# ==================================================================
DIAGNOSIS = {
    "loss 稳定卡在 ln(vocab_size)":        None,
    "loss 第 3 步就变成 nan":              None,
    "训练 loss 掉到 0.001，生成全是乱码":   None,
    "训练正常，推理输出全是同一个 token":    None,
    "第一个 epoch 正常，第二个 epoch OOM":  None,
}
# 可选病因（填字符串）:
#   "lr过大或fp16溢出"  "标签泄漏"  "模型没学到东西"  "忘了model.eval或no_grad"  "循环里存了tensor"


@task(8, "症状 -> 病因")
def t08():
    want = {
        "loss 稳定卡在 ln(vocab_size)": "模型没学到东西",
        "loss 第 3 步就变成 nan": "lr过大或fp16溢出",
        "训练 loss 掉到 0.001，生成全是乱码": "标签泄漏",
        "训练正常，推理输出全是同一个 token": "忘了model.eval或no_grad",
        "第一个 epoch 正常，第二个 epoch OOM": "循环里存了tensor",
    }
    wrong = [f"{k!r}: 你填 {v!r}, 应为 {want[k]!r}" for k, v in DIAGNOSIS.items() if v != want[k]]
    true(not wrong, "\n       " + "\n       ".join(wrong))


# ==================================================================
# 09  把新模块接进训练循环 —— 顺便揭一个反直觉的事实
# ==================================================================
def _toy_data():
    """4 个一维簇心 [-3, -1, 1, 3]，每个 16 个点，噪声 0.1。"""
    torch.manual_seed(42)
    c = torch.tensor([[-3.], [-1.], [1.], [3.]])
    return c.repeat_interleave(16, 0) + torch.randn(64, 1) * 0.1


def train_vq_toy(steps=300, lr=1e-1):
    """用 VQ 把 4 个簇心学出来。
      - data = _toy_data()                       (64, 1)
      - codebook = nn.Parameter(torch.linspace(-0.3, 0.3, 4).reshape(4, 1))
        （故意全都挤在原点附近，看它能不能自己散开到 4 个簇）
      - 每步:  zq, idx = vq_quantize(data, codebook)
               loss = F.mse_loss(zq, data) + vq_loss(data, codebook[idx], beta=0.0)
      - AdamW 只优化 codebook
    返回 loss 列表（float）。
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "把 VQ 接进训练循环")
def t09():
    losses = train_vq_toy()
    true(isinstance(losses[0], float), "要存 .item()")
    true(losses[0] > 5.0, f"初始 loss 应该很大（码本全在原点，数据在 ±3），实际 {losses[0]:.4f}")
    true(losses[-1] < 0.03,
         f"最终 loss = {losses[-1]:.5f}，应该 < 0.03（≈ 两倍的噪声方差）。"
         "降不下来就回去检查 vq_quantize 的直通和 vq_loss 的 stop-gradient")


@task(10, "反直觉: 重建项对码本没有梯度")
def t10():
    """这一题解释上一题为什么必须带 vq_loss。"""
    data = _toy_data()
    cb = nn.Parameter(torch.linspace(-0.3, 0.3, 4).reshape(4, 1))
    zq, idx = vq_quantize(data, cb)
    recon = F.mse_loss(zq, data)
    true(recon.requires_grad is False and recon.grad_fn is None,
         "直通估计器里的 (e - z).detach() 把码本这条路彻底切断了，"
         "所以重建项**根本给不了码本梯度** —— 码本只能靠 vq_loss 的 codebook 项更新。"
         "如果你这里 requires_grad=True，说明 detach 的位置写错了")
    # 而 vq_loss 的 codebook 项能给
    l = vq_loss(data, cb[idx], beta=0.0)
    l.backward()
    true(cb.grad is not None and cb.grad.abs().sum() > 0, "codebook 项应该给出非零梯度")


# ==================================================================
# 11  毕业自评
# ==================================================================
# 全部做到之后改成 True:
#   [ ] 前 13 天的 drills 全部跑到 [OK]
#   [ ] 能在不看任何参考的情况下默写出 Transformer Block
#   [ ] 能说清 DDPM 和 Flow Matching 在代码上差在哪几行
#   [ ] 挑了一篇本课没讲过的论文（VQ-VAE / Consistency Models / Mamba /
#       SigLIP / Whisper encoder 任选），4 小时内跑通了 smoke test
GRADUATED = False


@task(11, "毕业自评")
def t11():
    true(GRADUATED, "四条都做到了再改。前三条是复习，第四条才是毕业考。")


if __name__ == "__main__":
    raise SystemExit(run("Day 14 · 元技能与毕业考"))
