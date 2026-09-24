"""Day 13 · 三条线在这里合流：音频 codec + LM —— 11 题

运行:  python drills/day13.py
今天你会把 Day 4/5 的 GPT 原封不动拿来生成音频。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, close, params

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 已给出：Day 4 的 Block，一个字都没改
# ==================================================================
class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


# ==================================================================
# 01  码率账 —— 音频"一秒钟等于多少 token"
# ==================================================================
def codec_bitrate(frame_rate, n_q, codebook_size):
    """每秒比特数 = frame_rate * n_q * log2(codebook_size)"""
    # --- TODO 01 ---
    raise todo()


def tokens_per_second(frame_rate, n_q):
    """LM 视角: 一秒音频等于多少个 token。"""
    # --- TODO 01b ---
    raise todo()


@task(1, "codec 码率")
def t01():
    # EnCodec 24kHz: 75 帧/秒, 8 个码本, 每本 1024
    true(close(codec_bitrate(75, 8, 1024), 6000.0), f"应为 6000 bps, 你算出 {codec_bitrate(75, 8, 1024)}")
    true(tokens_per_second(75, 8) == 600,
         "一秒音频 = 600 个 token。对比一下：一秒说话大约 3 个汉字。"
         "这就是音频 LM 上下文吃紧的原因")


# ==================================================================
# 02  离散码的形状
# ==================================================================
def code_shape(seconds, frame_rate, n_q):
    """返回 (n_q, T) 这个 tuple，T = int(seconds * frame_rate)。"""
    # --- TODO 02 ---
    raise todo()


@task(2, "codes 的形状")
def t02():
    true(code_shape(2.0, 75, 8) == (8, 150))
    true(code_shape(0.5, 50, 4) == (4, 25))


# ==================================================================
# 03  delay pattern：多码本自回归的关键技巧
# ==================================================================
def build_delay(codes, pad_id):
    """codes: (B, n_q, T) -> (B, n_q, T + n_q - 1)
    第 k 个码本整体右移 k 步，空出来的位置填 pad_id。
    这样第 t 步预测时，第 k 个码本能看到第 k-1 个码本同一帧的结果。
    """
    # --- TODO 03 ---
    raise todo()


@task(3, "构造 delay pattern")
def t03():
    c = torch.tensor([[[1, 2, 3], [4, 5, 6]]])       # (1, 2, 3)
    d = build_delay(c, pad_id=0)
    shape_is(d, (1, 2, 4))
    eq(d[0, 0], torch.tensor([1, 2, 3, 0]), msg="第 0 个码本不移动，尾部补 pad")
    eq(d[0, 1], torch.tensor([0, 4, 5, 6]), msg="第 1 个码本右移 1 步，头部补 pad")


# ==================================================================
# 04  还原 delay —— 往返必须无损
# ==================================================================
def undo_delay(delayed, T):
    """build_delay 的逆操作: (B, n_q, T + n_q - 1) -> (B, n_q, T)"""
    # --- TODO 04 ---
    raise todo()


@task(4, "delay pattern 往返")
def t04():
    seed(0)
    for n_q in [1, 2, 4, 8]:
        c = torch.randint(1, 100, (3, n_q, 20))
        eq(undo_delay(build_delay(c, 0), 20), c, msg=f"n_q={n_q} 时往返不一致")


# ==================================================================
# 05  多码本嵌入：n_q 张表求和
# ==================================================================
class CodeEmbedding(nn.Module):
    """n_q 个 nn.Embedding(vocab, dim)，forward(codes) 把每个码本查表后**相加**。
    codes: (B, n_q, T) -> (B, T, dim)
    """

    def __init__(self, n_q, vocab, dim):
        super().__init__()
        # --- TODO 05 ---
        raise todo()


@task(5, "多码本嵌入求和")
def t05():
    seed(1)
    m = CodeEmbedding(4, 50, 32)
    codes = torch.randint(0, 50, (2, 4, 9))
    out = m(codes)
    shape_is(out, (2, 9, 32))
    true(params(m) == 4 * 50 * 32, f"应该是 4 张独立的表: {params(m)}")
    # 换掉任意一个码本的内容，输出都必须变
    for k in range(4):
        c2 = codes.clone()
        c2[:, k] = (c2[:, k] + 1) % 50
        true(not torch.allclose(m(c2), out), f"改了第 {k} 个码本输出没变 —— 那张表没用上")


# ==================================================================
# 06  n_q 个输出头
# ==================================================================
class MultiHead(nn.Module):
    """一个 Linear(dim, n_q * vocab)，输出 reshape 成 (B, n_q, T, vocab)。"""

    def __init__(self, dim, n_q, vocab):
        super().__init__()
        # --- TODO 06 ---
        raise todo()


@task(6, "n_q 个输出头")
def t06():
    m = MultiHead(32, 4, 50)
    shape_is(m(torch.randn(2, 9, 32)), (2, 4, 9, 50))


# ==================================================================
# 07  多码本的 loss
# ==================================================================
def codebook_loss(logits, targets, ignore_index=-100):
    """logits: (B, n_q, T, V)   targets: (B, n_q, T)
    对所有码本一起算交叉熵（先展平成 (B*n_q*T, V)），ignore_index 用于跳过 delay 补出来的位置。
    """
    # --- TODO 07 ---
    raise todo()


@task(7, "多码本 loss")
def t07():
    seed(2)
    logits = torch.randn(2, 4, 5, 50)
    tgt = torch.randint(0, 50, (2, 4, 5))
    ref = F.cross_entropy(logits.reshape(-1, 50), tgt.reshape(-1))
    eq(codebook_loss(logits, tgt), ref)
    # 被忽略的位置不能参与
    tgt2 = tgt.clone()
    tgt2[:, :, 0] = -100
    ref2 = F.cross_entropy(logits[:, :, 1:].reshape(-1, 50), tgt[:, :, 1:].reshape(-1))
    eq(codebook_loss(logits, tgt2), ref2, msg="ignore_index 没生效")


# ==================================================================
# 08  完整的 audio LM —— 就是 Day 4 的 GPT 换了输入输出
# ==================================================================
class AudioLM(nn.Module):
    """结构:
        emb  = CodeEmbedding(n_q, vocab, dim)
        pos  = nn.Embedding(max_len, dim)
        h    = n_layer 个 Block(dim, n_head)     <-- 顶部给出的那个，一个字没改
        ln_f = LayerNorm(dim)
        head = MultiHead(dim, n_q, vocab)
    forward(codes) -> (B, n_q, T, vocab)
    """

    def __init__(self, n_q=4, vocab=50, dim=64, n_layer=2, n_head=4, max_len=256):
        super().__init__()
        # --- TODO 08 ---
        raise todo()


@task(8, "AudioLM 前向")
def t08():
    seed(3)
    m = AudioLM(n_q=4, vocab=50, dim=64, n_layer=2)
    codes = torch.randint(0, 50, (2, 4, 16))
    out = m(codes)
    shape_is(out, (2, 4, 16, 50))
    # 因果性：改了后面的 token，前面的预测不能变
    c2 = codes.clone()
    c2[:, :, 8:] = torch.randint(0, 50, (2, 4, 8))
    eq(m(codes)[:, :, :8], m(c2)[:, :, :8], tol=1e-4,
       msg="因果性挂了 —— Block 里的 is_causal 是不是丢了？")
    # 初始 loss 应该在 ln(V) 附近
    loss = codebook_loss(out, codes)
    true(abs(loss.item() - math.log(50)) < 1.5,
         f"初始 loss={loss.item():.2f}, 应该在 ln(50)={math.log(50):.2f} 附近")


# ==================================================================
# 09  文本条件：和 Day 9 的 cross-attn 一模一样
# ==================================================================
def text_cross_attn(audio_h, text_emb, wq, wk, wv, n_head):
    """audio_h: (B, Ta, C)   text_emb: (B, Tt, C)
    Q 来自音频，KV 来自文本。返回 (B, Ta, C)。
    对照一下 Day 9 的 cross_attention —— 除了变量名，代码完全一样。
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "文本条件 cross-attn")
def t09():
    seed(4)
    B, Ta, Tt, C, nh = 2, 12, 5, 32, 4
    a, t = torch.randn(B, Ta, C), torch.randn(B, Tt, C)
    wq, wk, wv = (torch.randn(C, C) * .1 for _ in range(3))
    out = text_cross_attn(a, t, wq, wk, wv, nh)
    shape_is(out, (B, Ta, C), "长度跟着音频走")
    q = (a @ wq).view(B, Ta, nh, C // nh).transpose(1, 2)
    k = (t @ wk).view(B, Tt, nh, C // nh).transpose(1, 2)
    v = (t @ wv).view(B, Tt, nh, C // nh).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Ta, C)
    eq(out, ref, tol=1e-4)


# ==================================================================
# 10  生成一步：n_q 个码本同时采
# ==================================================================
@torch.no_grad()
def generate_codes(model, prompt, n_new, vocab):
    """prompt: (1, n_q, T0) 已有的码。每步:
        1) logits = model(当前序列)
        2) 取最后一个时间步的 logits: (1, n_q, vocab)
        3) 每个码本各取 argmax
        4) 拼到序列末尾
    返回 (1, n_q, T0 + n_new)。
    """
    # --- TODO 10 ---
    raise todo()


@task(10, "多码本生成")
def t10():
    seed(5)
    m = AudioLM(n_q=4, vocab=50, dim=64, n_layer=1).eval()
    p = torch.randint(0, 50, (1, 4, 6))
    out = generate_codes(m, p, 5, 50)
    shape_is(out, (1, 4, 11))
    eq(out[:, :, :6], p, msg="前缀被改了")
    true(out.dtype == torch.long and out.max() < 50 and out.min() >= 0)
    eq(out, generate_codes(m, p, 5, 50), msg="贪心生成必须确定性")


# ==================================================================
# 11  合流自评 —— 今天唯一真正重要的一题
# ==================================================================
# 逐条确认，全部为真之后把 CONVERGED 改成 True:
#   [ ] 本文件顶部的 Block，和你 Day 4 写的 CausalSelfAttention+MLP 是同一段代码
#   [ ] AudioLM 和 GPT 的差别只有: 输入是 n_q 张嵌入表求和、输出是 n_q 个头
#   [ ] text_cross_attn 和 Day 9 图像那边的 cross_attention 除了变量名完全一样
#   [ ] 所以 LLM / 文生图 / 文生音 三条线，最后落到代码上是同一个 Block
CONVERGED = False


@task(11, "三线合流（自评）")
def t11():
    true(CONVERGED, "逐条确认上面四句话。这四句话就是这两周的全部结论。")


if __name__ == "__main__":
    raise SystemExit(run("Day 13 · 音频 codec 与 Audio LM"))
