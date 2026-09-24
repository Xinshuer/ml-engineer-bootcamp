"""Day 04 · nanoGPT，一天手写一个 Transformer —— 13 题

运行:  python drills/day04.py
这是全课最重要的一天。第 13 题（默写）不跑代码，但请务必做。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, params

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 01  causal mask
# ==================================================================
def causal_bias(T, device=None):
    """返回 (1, 1, T, T) 的加性 mask：允许的位置 0.0，禁止的位置 -inf。
    位置 t 只能看到 <= t 的位置。用 torch.tril。
    """
    # --- TODO 01 ---
    raise todo()


@task(1, "causal mask 的方向")
def t01():
    m = causal_bias(4)
    shape_is(m, (1, 1, 4, 4))
    true(m[0, 0, 0, 0] == 0, "对角线（看自己）必须允许")
    true(m[0, 0, 0, 1] == float("-inf"), "第 0 个位置不能看到第 1 个 —— 你的 mask 反了")
    true(m[0, 0, 3, 0] == 0, "第 3 个位置应该能看到第 0 个")
    true(torch.isinf(m).sum() == 6, f"应该正好屏蔽 6 个位置，你屏蔽了 {int(torch.isinf(m).sum())} 个")


# ==================================================================
# 02  手写 attention，和 PyTorch 内置的对拍
# ==================================================================
def attention(q, k, v, causal=True):
    """q, k, v: (B, nh, T, hs)  ->  (B, nh, T, hs)
    自己写 scores -> mask -> softmax -> @v。不许调 F.scaled_dot_product_attention。
    """
    # --- TODO 02 ---
    raise todo()


@task(2, "手写 attention 对拍 SDPA")
def t02():
    seed(0)
    q, k, v = (torch.randn(2, 3, 6, 8) for _ in range(3))
    eq(attention(q, k, v, causal=True), F.scaled_dot_product_attention(q, k, v, is_causal=True))
    eq(attention(q, k, v, causal=False), F.scaled_dot_product_attention(q, k, v, is_causal=False))


# ==================================================================
# 03  因果性的硬检验 —— 改未来不能影响过去
# ==================================================================
@task(3, "因果性检验")
def t03():
    seed(1)
    q, k, v = (torch.randn(1, 2, 6, 4) for _ in range(3))
    out1 = attention(q, k, v, causal=True)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 4:] = torch.randn(1, 2, 2, 4)     # 只改后两个时间步
    v2[:, :, 4:] = torch.randn(1, 2, 2, 4)
    out2 = attention(q, k2, v2, causal=True)
    eq(out1[:, :, :4], out2[:, :, :4],
       msg="改了未来的 k/v，前 4 个位置的输出也变了 —— 信息泄漏了")
    true(not torch.allclose(out1[:, :, 4:], out2[:, :, 4:]), "后两个位置应该变了才对")


# ==================================================================
# 04  CausalSelfAttention 模块
# ==================================================================
class CausalSelfAttention(nn.Module):
    """要求:
      - self.qkv:  nn.Linear(C, 3C, bias=False)
      - self.proj: nn.Linear(C, C,  bias=False)
      - forward(x): (B,T,C) -> (B,T,C)，内部拆头、算 causal attention、合并头、过 proj
      - 用 register_buffer 存 mask（名字叫 "bias"），不要每次 forward 现造
    """

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        # --- TODO 04 ---
        raise todo()


@task(4, "CausalSelfAttention 模块")
def t04():
    seed(2)
    m = CausalSelfAttention(32, 4, 16)
    x = torch.randn(2, 9, 32)
    shape_is(m(x), (2, 9, 32))
    true("bias" in dict(m.named_buffers()), "mask 应该注册成 buffer，不是每次现算")
    true(params(m) == 32 * 32 * 3 + 32 * 32, f"参数量不对: {params(m)}（qkv 无 bias + proj 无 bias）")
    # 变长输入也要能跑（T < block_size）
    shape_is(m(torch.randn(1, 3, 32)), (1, 3, 32))
    # 因果性
    x2 = x.clone()
    x2[:, 5:] = torch.randn(2, 4, 32)
    eq(m(x)[:, :5], m(x2)[:, :5], msg="模块级别的因果性挂了")


# ==================================================================
# 05  MLP
# ==================================================================
class MLP(nn.Module):
    """标准 GPT-2 MLP: Linear(C, 4C) -> GELU -> Linear(4C, C)，两层都带 bias。"""

    def __init__(self, n_embd):
        super().__init__()
        # --- TODO 05 ---
        raise todo()


@task(5, "MLP")
def t05():
    m = MLP(16)
    shape_is(m(torch.randn(2, 5, 16)), (2, 5, 16))
    true(params(m) == 16 * 64 + 64 + 64 * 16 + 16, f"参数量不对: {params(m)}")


# ==================================================================
# 06  Block：pre-LN 残差
# ==================================================================
class Block(nn.Module):
    """x = x + attn(ln1(x));  x = x + mlp(ln2(x))
    注意是 pre-LN（LN 在残差分支里面），不是 post-LN。
    """

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        # --- TODO 06 ---
        raise todo()


@task(6, "Block: pre-LN 残差")
def t06():
    seed(3)
    b = Block(32, 4, 16)
    x = torch.randn(2, 7, 32)
    shape_is(b(x), (2, 7, 32))
    # 把两个子层的输出权重清零，Block 就该退化成恒等映射 —— 这能同时验证残差和 LN 的位置
    with torch.no_grad():
        b.attn.proj.weight.zero_()
        for p in b.mlp.parameters():
            p.zero_()
    eq(b(x), x, tol=1e-5,
       msg="子层输出清零后 Block 应该是恒等映射。不相等说明残差写错了（比如写成了 x = attn(ln(x))）")


# ==================================================================
# 07  完整 GPT
# ==================================================================
class GPT(nn.Module):
    """要求:
      - wte: nn.Embedding(vocab_size, C)
      - wpe: nn.Embedding(block_size, C)
      - h:   nn.ModuleList of n_layer 个 Block
      - ln_f: nn.LayerNorm(C)
      - lm_head: nn.Linear(C, vocab_size, bias=False)，且与 wte **权重共享**
      - 所有 Linear / Embedding 的权重用 normal_(mean=0, std=0.02) 初始化，bias 清零
        （nanoGPT 的做法。不这么初始化，第 7 题的初始 loss 会远大于 ln(V)）
      - forward(idx, targets=None) -> (logits, loss)。targets 为 None 时 loss 是 None。
    """

    def __init__(self, vocab_size, block_size, n_layer=2, n_head=4, n_embd=32):
        super().__init__()
        # --- TODO 07 ---
        raise todo()

    def forward(self, idx, targets=None):
        raise todo()


@task(7, "GPT forward + loss")
def t07():
    seed(4)
    m = GPT(vocab_size=17, block_size=16, n_layer=2)
    idx = torch.randint(0, 17, (2, 9))
    logits, loss = m(idx)
    shape_is(logits, (2, 9, 17))
    true(loss is None, "没给 targets 时 loss 应该是 None")
    logits, loss = m(idx, idx)
    true(loss.dim() == 0, "loss 要是标量")
    # 随机初始化的模型，loss 应该在 ln(V) 附近
    true(abs(loss.item() - math.log(17)) < 0.7,
         f"初始 loss = {loss.item():.3f}，应该在 ln(17)={math.log(17):.3f} 附近。"
         "偏太多说明初始化或 loss 展平写错了")


# ==================================================================
# 08  weight tying
# ==================================================================
@task(8, "weight tying")
def t08():
    m = GPT(vocab_size=17, block_size=16, n_layer=1)
    true(m.lm_head.weight is m.wte.weight,
         "lm_head 和 wte 必须是同一个张量对象（is，不是 equal）")
    ids = [id(p) for p in m.parameters()]
    true(len(ids) == len(set(ids)), "共享的那个权重被 parameters() 数了两遍")


# ==================================================================
# 09  参数量公式 —— 面试和估算显存都要用
# ==================================================================
def gpt_param_count(V, C, L, block_size, tied=True):
    """按下面的结构手算参数量（不许构造模型去数）:
       wte: V*C     wpe: block_size*C
       每个 block:  ln1(2C) + qkv(3C*C) + proj(C*C) + ln2(2C) + fc1(4C*C+4C) + fc2(4C*C+C)
       末尾: ln_f(2C)
       tied=True 时 lm_head 不额外计数。
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "参数量公式")
def t09():
    for V, C, L, bs in [(17, 32, 2, 16), (65, 64, 3, 32), (1000, 128, 4, 64)]:
        m = GPT(V, bs, n_layer=L, n_head=C // 8, n_embd=C)
        real = sum(p.numel() for p in set(m.parameters()))
        got = gpt_param_count(V, C, L, bs)
        true(got == real, f"V={V},C={C},L={L}: 你算 {got}, 实际 {real}, 差 {got-real}")


# ==================================================================
# 10  贪心生成
# ==================================================================
@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens, block_size):
    """每步取 argmax 追加。序列超过 block_size 时只喂最后 block_size 个。
    返回 (B, T0+max_new_tokens)。
    """
    # --- TODO 10 ---
    raise todo()


@task(10, "贪心生成 + 上下文裁剪")
def t10():
    seed(5)
    m = GPT(vocab_size=11, block_size=8, n_layer=1).eval()
    idx = torch.randint(0, 11, (2, 3))
    out = generate_greedy(m, idx, 10, 8)
    shape_is(out, (2, 13), "生成后的总长度不对")
    eq(out[:, :3], idx, msg="前缀被改掉了")
    true(out.max() < 11 and out.min() >= 0)
    eq(out, generate_greedy(m, idx, 10, 8), msg="贪心生成必须是确定性的")
    # 超长时不能崩（这就是必须裁剪上下文的原因）
    long_idx = torch.randint(0, 11, (1, 20))
    shape_is(generate_greedy(m, long_idx, 3, 8), (1, 23))


# ==================================================================
# 11  找 bug —— 这个 attention 有 3 处错
# ==================================================================
def attention_buggy(q, k, v):
    T = q.size(-2)
    att = q @ k.transpose(-2, -1)
    mask = torch.triu(torch.ones(T, T))
    att = att.masked_fill(mask == 0, float("-inf"))
    att = F.softmax(att, dim=-2)
    return att @ v


@task(11, "找 bug: attention_buggy")
def t11():
    seed(6)
    q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
    eq(attention_buggy(q, k, v), F.scaled_dot_product_attention(q, k, v, is_causal=True),
       msg="三处: 缩放因子 / mask 的三角方向 / softmax 的维度")


# ==================================================================
# 12  找 bug —— 这个 Block 有 2 处错
# ==================================================================
class BlockBuggy(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.mlp(self.ln2(x))
        return x


@task(12, "找 bug: BlockBuggy")
def t12():
    seed(7)
    b = BlockBuggy(32, 4, 16)
    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        b.attn.proj.weight.zero_()
        for p in b.mlp.parameters():
            p.zero_()
    eq(b(x), x, tol=1e-5, msg="两处: LN 应该在残差分支内部(pre-LN)；MLP 那一行漏了残差")


# ==================================================================
# 13  默写清单 —— 不跑代码，但这是今天最该做的一题
# ==================================================================
# 关掉所有参考，在一个空文件里默写 Block 类。写完对照下面每一条自查，
# 全部打勾之后把 DICTATION_DONE 改成 True。
#
#   [ ] class Block(nn.Module) 且 __init__ 里第一行是 super().__init__()
#   [ ] ln1 / attn / ln2 / mlp 四个子模块都建了
#   [ ] forward 里是 x = x + attn(ln1(x))，不是 x = attn(ln1(x))
#   [ ] mlp 那一行同样带残差
#   [ ] attention 里 qkv 一次投影成 3C 再 split
#   [ ] view 成 (B,T,nh,hs) 之后 transpose(1,2)
#   [ ] scores 除以 sqrt(head_dim)
#   [ ] mask 用 tril，masked_fill 成 -inf
#   [ ] softmax 在最后一维
#   [ ] 合并头时 transpose 回来要 .contiguous() 再 view
#   [ ] 最后过一层 proj

DICTATION_DONE = False


@task(13, "默写 Block（自评）")
def t13():
    true(DICTATION_DONE, "还没默写。这一题比前面 12 题加起来都重要，别跳。")


if __name__ == "__main__":
    raise SystemExit(run("Day 04 · nanoGPT"))
