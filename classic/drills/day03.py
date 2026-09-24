"""Day 03 · 文本进模型之前发生了什么 —— 11 题

运行:  python drills/day03.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, close

import math
import numpy as np
import torch


# ==================================================================
# 01  char-level tokenizer —— 20 行的完整 tokenizer
# ==================================================================
class CharTokenizer:
    """用法:
        tok = CharTokenizer("hello world")
        tok.vocab_size      -> 去重后的字符数
        tok.encode("hell")  -> [int, int, int, int]
        tok.decode([...])   -> "hell"
    要求: 字典序建表（sorted(set(text))），保证 decode(encode(s)) == s。
    """

    def __init__(self, text):
        # --- TODO 01 ---
        raise todo()


@task(1, "CharTokenizer 往返")
def t01():
    txt = "the quick brown fox jumps over the lazy dog"
    tok = CharTokenizer(txt)
    true(tok.vocab_size == len(set(txt)), f"vocab_size 不对: {tok.vocab_size}")
    ids = tok.encode("quick fox")
    true(all(isinstance(i, int) for i in ids), "encode 要返回 int 列表")
    true(tok.decode(ids) == "quick fox", "往返不一致")
    true(tok.encode(" ") == tok.encode(" "), "同一个字符要映射到同一个 id")
    true(min(ids) >= 0 and max(ids) < tok.vocab_size, "id 越界了")


# ==================================================================
# 02  BPE 的一步：统计相邻 pair 频次
# ==================================================================
def pair_counts(tokens):
    """tokens: list[str]。统计所有相邻二元组出现次数。
    ["a","b","a","b","c"] -> {("a","b"): 2, ("b","a"): 1, ("b","c"): 1}
    """
    # --- TODO 02 ---
    raise todo()


def merge_pair(tokens, pair):
    """把 tokens 里所有连续出现的 pair 合并成一个 token（字符串拼接）。
    从左往右扫，合并过的位置不再参与下一次合并。
    (["a","b","a","b"], ("a","b")) -> ["ab", "ab"]
    """
    # --- TODO 02b ---
    raise todo()


@task(2, "BPE: 统计 pair 与合并")
def t02():
    t = list("abababc")
    c = pair_counts(t)
    true(c[("a", "b")] == 3, f"('a','b') 应该出现 3 次, 你数出 {c.get(('a','b'))}")
    true(c[("b", "a")] == 2)
    true(max(c, key=c.get) == ("a", "b"))
    true(merge_pair(t, ("a", "b")) == ["ab", "ab", "ab", "c"], f"{merge_pair(t, ('a','b'))}")
    # 重叠情况: "aaa" 合并 ("a","a") 只能合出一个 "aa"，剩一个 "a"
    true(merge_pair(list("aaa"), ("a", "a")) == ["aa", "a"], "重叠 pair 处理错了")


# ==================================================================
# 03  压缩率 —— 为什么中文更"贵"
# ==================================================================
def compression_ratio(text, encode_fn):
    """返回 字符数 / token 数。越大说明这个 tokenizer 对这段文本越高效。"""
    # --- TODO 03 ---
    raise todo()


@task(3, "压缩率计算")
def t03():
    fake = lambda s: list(range(len(s) // 2))     # 假 tokenizer: 两个字符一个 token
    true(close(compression_ratio("abcdefgh", fake), 2.0))
    true(close(compression_ratio("abcd", lambda s: [1, 2, 3, 4]), 1.0))


# ==================================================================
# 04  get_batch —— 全课最常写的一个函数
# ==================================================================
def get_batch(data, batch_size, block_size, generator=None):
    """data: 1D LongTensor（整段语料的 token id）
    随机取 batch_size 个起点 i，返回:
        x = data[i   : i+block_size]
        y = data[i+1 : i+1+block_size]
    形状都是 (batch_size, block_size)，dtype int64。
    起点必须保证 y 不越界。generator 传给 torch.randint 用于复现。
    """
    # --- TODO 04 ---
    raise todo()


@task(4, "get_batch 与 x/y 错位")
def t04():
    data = torch.arange(1000)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, 4, 8, g)
    shape_is(x, (4, 8))
    shape_is(y, (4, 8))
    true(x.dtype == torch.long and y.dtype == torch.long, "dtype 必须是 int64")
    eq(y, x + 1, msg="y 必须是 x 右移一位。这一行写错，模型会学会抄答案，loss 掉到 0 但生成全是垃圾")
    # 越界检查: 用最短的合法数据反复采样
    short = torch.arange(9)
    for _ in range(50):
        a, b = get_batch(short, 2, 8)
        true(int(b.max()) <= 8, "y 越界了 —— 起点上界应该是 len(data)-block_size-1")


# ==================================================================
# 05  memmap 打包 —— 语料大到装不进内存时的标准做法
# ==================================================================
def save_bin(path, ids):
    """把 ids(list[int] 或 1D ndarray) 以 uint16 存成二进制文件。"""
    # --- TODO 05 ---
    raise todo()


def load_bin(path):
    """用 np.memmap 以只读方式加载，返回 memmap 对象（不要整个读进内存）。"""
    # --- TODO 05b ---
    raise todo()


@task(5, "np.memmap 存取")
def t05():
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "train.bin")
    ids = list(range(1000))
    save_bin(p, ids)
    m = load_bin(p)
    true(isinstance(m, np.memmap), "要返回 np.memmap，不是普通 ndarray")
    true(m.dtype == np.uint16, f"dtype 应为 uint16, 实际 {m.dtype}")
    true(len(m) == 1000)
    true(int(m[999]) == 999)
    true(os.path.getsize(p) == 2000, "uint16 每个 token 占 2 字节，1000 个应该是 2000 字节")


# ==================================================================
# 06  bigram 的转移计数 —— 你的第一个"模型"
# ==================================================================
def bigram_counts(ids, vocab_size):
    """ids: 1D LongTensor。返回 (V, V) 的计数矩阵 C，C[i, j] = i 后面紧跟 j 的次数。
    不许用 python 循环遍历 ids（用 index_put_ 或 bincount 或 scatter_add_）。
    """
    # --- TODO 06 ---
    raise todo()


@task(6, "bigram 转移计数（向量化）")
def t06():
    ids = torch.tensor([0, 1, 0, 1, 2])
    C = bigram_counts(ids, 3)
    shape_is(C, (3, 3))
    eq(C, torch.tensor([[0., 2., 0.], [1., 0., 1.], [0., 0., 0.]]))
    big = torch.randint(0, 50, (20000,))
    true(bigram_counts(big, 50).sum().item() == 19999, "总计数应该是 len(ids)-1")


# ==================================================================
# 07  ln(V) 直觉 —— 每天都要用它判断"是不是写错了"
# ==================================================================
def uniform_loss(vocab_size):
    """一个什么都没学到的模型，交叉熵应该是多少？返回这个数。"""
    # --- TODO 07 ---
    raise todo()


def is_broken(loss, vocab_size, tol=0.05):
    """训练了半天 loss 还在 uniform_loss 附近（相对误差 < tol）就返回 True。"""
    # --- TODO 07b ---
    raise todo()


@task(7, "ln(V) 基线")
def t07():
    true(close(uniform_loss(65), math.log(65)))
    true(is_broken(4.17, 65) is True, "4.17 ≈ ln(65)，这就是没学到东西")
    true(is_broken(1.48, 65) is False)


# ==================================================================
# 08  padding 批次的 mask 与 position ids
# ==================================================================
def pad_batch(seqs, pad_id=0):
    """seqs: list[list[int]]，返回 (ids, attn_mask, pos_ids)，都是 (B, Tmax) int64。
    - ids: 右侧补 pad_id
    - attn_mask: 有效位 1，pad 位 0
    - pos_ids: 每个样本从 0 开始数**有效位**的位置；pad 位置填 0
    """
    # --- TODO 08 ---
    raise todo()


@task(8, "padding + mask + position ids")
def t08():
    ids, m, pos = pad_batch([[5, 6, 7], [8], [9, 10]], pad_id=-1)
    eq(ids, torch.tensor([[5, 6, 7], [8, -1, -1], [9, 10, -1]]))
    eq(m, torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]]))
    eq(pos, torch.tensor([[0, 1, 2], [0, 0, 0], [0, 1, 0]]))
    true(ids.dtype == torch.long)


# ==================================================================
# 09  把 mask 变成 attention 用的加性偏置
# ==================================================================
def mask_to_bias(attn_mask):
    """attn_mask: (B, T) 的 0/1  ->  (B, 1, 1, T) 的加性偏置:
    有效位 0.0，pad 位 -inf。加到 scores 上再 softmax，pad 位的权重就会变成 0。
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "mask -> 加性 -inf 偏置")
def t09():
    m = torch.tensor([[1, 1, 0]])
    b = mask_to_bias(m)
    shape_is(b, (1, 1, 1, 3))
    true(b[0, 0, 0, 0].item() == 0.0)
    true(b[0, 0, 0, 2].item() == float("-inf"), "pad 位要用 -inf，不是 -1e9 之外的随便一个数")
    scores = torch.zeros(1, 1, 1, 3) + b
    w = scores.softmax(-1)
    eq(w, torch.tensor([[[[0.5, 0.5, 0.0]]]]), msg="加完偏置 softmax 后 pad 位应该正好是 0")


# ==================================================================
# 10  找 bug —— 这个 get_batch 有 3 处错
# ==================================================================
def get_batch_buggy(data, batch_size, block_size):
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i:i + block_size] for i in ix])
    return x.float(), y.float()


@task(10, "找 bug: get_batch_buggy")
def t10():
    data = torch.arange(200)
    for _ in range(30):
        x, y = get_batch_buggy(data, 4, 16)
        shape_is(x, (4, 16))
        true(x.dtype == torch.long and y.dtype == torch.long, "embedding 层只吃 int64")
        eq(y, x + 1, msg="y 没有右移")


# ==================================================================
# 11  shape 速算 —— 一个训练 step 里的所有张量
# ==================================================================
# 设 B=8, T=256, V=50257, C=768。填 tuple。
BATCH_SHAPES = {
    "x  (输入 token id)":        None,
    "y  (标签)":                 None,
    "tok_emb = wte(x)":          None,
    "pos_emb = wpe(pos)":        None,   # pos 是 (T,)
    "hidden (每个 block 的输出)": None,
    "logits = lm_head(hidden)":  None,
    "logits.view(-1, V)":        None,
    "y.view(-1)":                None,
    "loss":                      None,   # 标量用空 tuple ()
}


@task(11, "一个训练 step 的全部 shape")
def t11():
    B, T, V, C = 8, 256, 50257, 768
    want = {
        "x  (输入 token id)": (B, T),
        "y  (标签)": (B, T),
        "tok_emb = wte(x)": (B, T, C),
        "pos_emb = wpe(pos)": (T, C),
        "hidden (每个 block 的输出)": (B, T, C),
        "logits = lm_head(hidden)": (B, T, V),
        "logits.view(-1, V)": (B * T, V),
        "y.view(-1)": (B * T,),
        "loss": (),
    }
    wrong = [f"{k!r}: 你填 {v}, 应为 {want[k]}" for k, v in BATCH_SHAPES.items()
             if v is None or tuple(v) != want[k]]
    true(not wrong, "\n       " + "\n       ".join(wrong))


if __name__ == "__main__":
    raise SystemExit(run("Day 03 · tokenizer 与数据管线"))
