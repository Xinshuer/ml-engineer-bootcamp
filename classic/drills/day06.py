"""Day 06 · 读别人的代码，调别人的模型 —— 11 题

运行:  python drills/day06.py
第 8 题（梯度累积等价性）是工程题里最值钱的一道。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, params, trainable, close

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 01  LoRALinear
# ==================================================================
class LoRALinear(nn.Module):
    """包住一个已有的 nn.Linear，加上低秩旁路。
        y = base(x) + (x @ A^T @ B^T) * (alpha / r)
    要求:
      - self.base 是传进来的那个 Linear，**参数全部冻结**（requires_grad=False）
      - self.A: (r, in_features)，用 normal_(0, 0.01) 初始化
      - self.B: (out_features, r)，**初始化为全 0**
      - self.scaling = alpha / r
    """

    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        # --- TODO 01 ---
        raise todo()


@task(1, "LoRALinear 结构")
def t01():
    seed(0)
    base = nn.Linear(16, 8)
    m = LoRALinear(base, r=4, alpha=8)
    shape_is(m(torch.randn(2, 5, 16)), (2, 5, 8))
    true(close(m.scaling, 2.0), f"scaling 应为 alpha/r = 2.0, 你的 {m.scaling}")
    shape_is(m.A, (4, 16), "A")
    shape_is(m.B, (8, 4), "B")


# ==================================================================
# 02  B 初始化为 0 的意义：微调开始的那一刻，模型完全没变
# ==================================================================
@task(2, "初始等价性")
def t02():
    seed(1)
    base = nn.Linear(16, 8)
    x = torch.randn(3, 16)
    before = base(x).clone()
    m = LoRALinear(base, r=4, alpha=8)
    eq(m(x), before, msg="B 初始化不是 0 —— 微调刚开始模型输出就变了，等于白白破坏了预训练权重")


# ==================================================================
# 03  只有 A / B 可训练
# ==================================================================
@task(3, "冻结 base")
def t03():
    m = LoRALinear(nn.Linear(16, 8), r=4)
    true(trainable(m) == 4 * 16 + 8 * 4, f"可训练参数应为 A+B = 96, 实际 {trainable(m)}")
    true(params(m) == 16 * 8 + 8 + 96, f"总参数量不对: {params(m)}")
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    true(all("base" not in n for n in names), f"base 没冻住: {names}")


# ==================================================================
# 04  合并权重 —— 部署时 LoRA 不该有额外开销
# ==================================================================
def merge_lora(m: LoRALinear) -> nn.Linear:
    """把 LoRA 合并回一个普通 nn.Linear: W' = W + scaling * B @ A，bias 不变。
    返回一个新的 nn.Linear，输出必须和 m 完全一致。
    """
    # --- TODO 04 ---
    raise todo()


@task(4, "合并 LoRA 权重")
def t04():
    seed(2)
    m = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
    with torch.no_grad():                      # 造一个非零的 B，模拟训练过
        m.B.normal_(0, 0.5)
    x = torch.randn(3, 16)
    merged = merge_lora(m)
    true(isinstance(merged, nn.Linear), "要返回普通 nn.Linear")
    eq(merged(x), m(x), tol=1e-5, msg="合并后输出变了 —— 检查 B@A 的顺序和 scaling")
    true(params(merged) == 16 * 8 + 8, "合并后不该还带着 A/B")


# ==================================================================
# 05  参数量占比 —— 为什么 LoRA 能省
# ==================================================================
def lora_ratio(in_f, out_f, r):
    """返回 LoRA 参数量 / 原权重参数量（不含 bias）。"""
    # --- TODO 05 ---
    raise todo()


@task(5, "LoRA 参数量占比")
def t05():
    true(close(lora_ratio(4096, 4096, 8), 2 * 8 / 4096, 1e-9),
         f"你算出 {lora_ratio(4096, 4096, 8)}")
    true(lora_ratio(4096, 4096, 8) < 0.005, "r=8 时应该不到 0.5%")


# ==================================================================
# 06  SFT 的 labels：只在 assistant 段回传梯度
# ==================================================================
def build_sft_labels(input_ids, prompt_len, ignore_index=-100):
    """input_ids: (T,) 一整条 "prompt + answer" 的 token。
    返回 labels: (T,)，其中:
      - 前 prompt_len 个位置填 ignore_index（不算 loss）
      - 其余位置等于 input_ids 本身
    这个细节写错模型会变傻，而且**完全不报错**。
    """
    # --- TODO 06 ---
    raise todo()


@task(6, "SFT labels 的 -100 掩码")
def t06():
    ids = torch.tensor([5, 6, 7, 8, 9])
    lab = build_sft_labels(ids, prompt_len=3)
    eq(lab, torch.tensor([-100, -100, -100, 8, 9]))
    true(lab.dtype == torch.long)
    # 验证 -100 真的被 cross_entropy 忽略
    logits = torch.randn(5, 20)
    l_all = F.cross_entropy(logits, ids)
    l_ans = F.cross_entropy(logits, lab)
    l_manual = F.cross_entropy(logits[3:], ids[3:])
    eq(l_ans, l_manual, msg="-100 应该被完全忽略，且分母只数有效位")
    true(not close(l_all, l_ans), "掩码没起作用")


# ==================================================================
# 07  chat template
# ==================================================================
def apply_chat_template(messages, add_generation_prompt=True):
    """messages: [{"role": "user"/"assistant"/"system", "content": str}, ...]
    输出格式（每段结尾都有换行）:
        <|im_start|>user\\nHi<|im_end|>\\n<|im_start|>assistant\\nHello<|im_end|>\\n
    add_generation_prompt=True 时，末尾再追加 "<|im_start|>assistant\\n"（让模型接着写）。
    """
    # --- TODO 07 ---
    raise todo()


@task(7, "chat template 拼接")
def t07():
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    s = apply_chat_template(msgs, add_generation_prompt=False)
    true(s == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n",
         f"\n       你的: {s!r}")
    s2 = apply_chat_template([msgs[0]], add_generation_prompt=True)
    true(s2.endswith("<|im_start|>assistant\n"), f"\n       你的: {s2!r}")


# ==================================================================
# 08  梯度累积必须和大 batch 等价
# ==================================================================
def grad_accum_step(model, xs, ys, n_micro):
    """把 (N, ...) 的一个大 batch 切成 n_micro 份依次 forward/backward，
    使得累积出的梯度与"一次性喂整个大 batch"完全相同。
    返回累积完成后的梯度字典 {参数名: grad.clone()}。函数内部自己负责 zero_grad。
    提示: F.cross_entropy 默认对 batch 取平均，所以每份的 loss 要除以 n_micro。
    """
    # --- TODO 08 ---
    raise todo()


@task(8, "梯度累积 == 大 batch")
def t08():
    seed(3)
    model = nn.Linear(6, 4)
    xs, ys = torch.randn(32, 6), torch.randint(0, 4, (32,))

    model.zero_grad()
    F.cross_entropy(model(xs), ys).backward()
    ref = {n: p.grad.clone() for n, p in model.named_parameters()}

    got = grad_accum_step(model, xs, ys, n_micro=4)
    for n in ref:
        eq(got[n], ref[n], tol=1e-5,
           msg=f"参数 {n} 的梯度对不上。忘了除以 n_micro 的话会大 4 倍")


# ==================================================================
# 09  warmup + cosine 学习率
# ==================================================================
def get_lr(it, warmup_iters, max_iters, max_lr, min_lr):
    """三段:
      it < warmup_iters              : 从 0 线性升到 max_lr，公式 max_lr * (it+1)/warmup_iters
      warmup_iters <= it <= max_iters: 余弦从 max_lr 降到 min_lr
                                       coeff = 0.5 * (1 + cos(pi * progress))
                                       progress = (it - warmup) / (max_iters - warmup)
      it > max_iters                 : 恒为 min_lr
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "warmup + cosine schedule")
def t09():
    f = lambda it: get_lr(it, 100, 1000, 6e-4, 6e-5)
    true(close(f(0), 6e-6, 1e-9), f"第 0 步应为 max_lr/warmup, 你的 {f(0)}")
    true(close(f(99), 6e-4, 1e-9), "warmup 最后一步应该正好到 max_lr")
    true(close(f(100), 6e-4, 1e-9), "衔接点不该跳变")
    true(close(f(550), (6e-4 + 6e-5) / 2, 1e-7), "中点应该是两端的平均")
    true(close(f(1000), 6e-5, 1e-9))
    true(close(f(5000), 6e-5, 1e-9), "超出 max_iters 后应该恒定")


# ==================================================================
# 10  显存估算 —— 决定你的 batch 能开多大
# ==================================================================
def optimizer_memory_bytes(n_params, dtype_bytes=4, optimizer="adamw"):
    """只算"参数 + 梯度 + 优化器状态"，不算激活值。
      adamw: 参数 + 梯度 + 一阶动量 + 二阶动量 = 4 份
      sgd  : 参数 + 梯度 = 2 份
      sgd_momentum: 3 份
    返回字节数。
    """
    # --- TODO 10 ---
    raise todo()


@task(10, "优化器显存估算")
def t10():
    true(optimizer_memory_bytes(1_000_000_000) == 16_000_000_000,
         "1B 参数用 fp32 AdamW 需要 16GB —— 这就是为什么 7B 模型全量微调单卡跑不动")
    true(optimizer_memory_bytes(1_000_000, optimizer="sgd") == 8_000_000)
    true(optimizer_memory_bytes(1_000_000, optimizer="sgd_momentum") == 12_000_000)


# ==================================================================
# 11  找 bug —— 这个 LoRA 有 3 处错
# ==================================================================
class LoRABuggy(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(base.out_features, r) * 0.01)
        self.scaling = alpha * r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


@task(11, "找 bug: LoRABuggy")
def t11():
    seed(4)
    base = nn.Linear(16, 8)
    x = torch.randn(3, 16)
    before = base(x).clone()
    m = LoRABuggy(base, r=4, alpha=8)
    eq(m(x), before, msg="三处: B 必须初始化为 0 / scaling 是 alpha/r 不是 alpha*r / base 要冻结")
    true(trainable(m) == 4 * 16 + 8 * 4, f"base 没冻结，可训练参数 {trainable(m)}")


if __name__ == "__main__":
    raise SystemExit(run("Day 06 · LoRA 与训练工程"))
