# 三模态手写训练营 · 自测题库

配合 14 天计划使用的 **169 道可自测代码练习**。每天一个文件，改完跑一下就知道对不对。

## 怎么用

```bash
cd G:/个人项目/ai-14days

python check.py              # 看 14 天总进度
python drills/day02.py       # 做 Day 02 的题
python drills/day02.py 05    # 只跑第 5 题
```

每个文件里的题长这样：

```python
def my_softmax(x, dim=-1):
    """不许用 F.softmax。必须减去最大值，否则大数会溢出。"""
    # --- TODO 05 ---
    raise todo()
```

把 `raise todo()` 换成你的实现，跑到全 `[OK]` 为止。

- `[OK]` 过了
- `[--]` 还没写
- `[XX]` 写了但不对（下面会打印哪里不对）

看完整报错栈：`set DRILL_TB=1` 再跑（PowerShell 里是 `$env:DRILL_TB=1`）。

## 卡住了怎么办

先自己想 5 分钟，再看 `solutions/dayNN.py`。参考答案带注释，尤其标了「修 1 / 修 2」的地方就是那道找 bug 题的答案。

想确认某道题**本身**没问题，用参考答案跑一遍：

```bash
python check.py 08 --sol
```

## 题目分布

| | 天 | 题数 | 内容 |
|---|---|---|---|
| 地基 | 01 | 12 | Python 的那 20%：dataclass、class、装饰器、pathlib |
| | 02 | 15 | shape 速算、手写 Linear/softmax/CE/LayerNorm、einsum、autograd、collate |
| LLM | 03 | 11 | tokenizer、BPE 合并、get_batch、memmap、bigram、ln(V) 基线 |
| | 04 | 13 | causal mask、attention 对拍 SDPA、Block、GPT、weight tying、参数量、生成 |
| | 05 | 13 | RMSNorm、RoPE（两种写法 + 相对位置性质）、SwiGLU、GQA、KV-Cache、采样 |
| | 06 | 11 | LoRA、权重合并、SFT 的 -100 掩码、梯度累积等价性、warmup+cosine、显存估算 |
| 图像 | 07 | 11 | 卷积尺寸、重参数化、KL 闭式 vs 蒙特卡洛、Encoder/Decoder、压缩比、后验坍塌 |
| | 08 | 14 | beta 调度、q_sample、**闭式 == 逐步加噪**、时间嵌入、FiLM、ResBlock、UNet、p_sample |
| | 09 | 12 | cross-attention、CFG、条件 dropout、**DDIM 精确还原**、latent diffusion 形状账 |
| | 10 | 12 | patchify、2D 位置编码、adaLN-Zero、DiTBlock、Rectified Flow、欧拉采样 |
| 音频 | 11 | 12 | 采样率算术、STFT 帧数、mel 刻度、log-mel、分帧、**相位丢失实验** |
| | 12 | 11 | length regulator、duration 对齐、teacher forcing、stop token、带 mask 的 loss |
| | 13 | 11 | codec 码率、**delay pattern 往返**、多码本嵌入/输出头、AudioLM、三线合流 |
| 元技能 | 14 | 11 | 五问法、smoke test、过拟合单 batch、梯度裁剪、**VQ（全新内容）**、症状→病因 |

加粗的是每天最该做通的那一题。

## 三类题，各有各的用处

1. **实现题** —— 大多数。写函数/模块，跟 PyTorch 内置或数学性质对拍。
2. **找 bug 题** —— 每天最后一道左右。给你一段有 2–4 处错的代码，改到测试过。错的都是真实世界里最常犯的：mask 反了、忘了 `zero_grad`、`exp(logvar)` 写成了 std、CFG 方向错、y 忘了右移。
3. **脑算题 / 自评题** —— 填 shape 的 dict、填诊断表、默写清单。不许先跑代码，先猜再验。

## 依赖

只需要 `torch`（`day11` 用到 `torch.stft`，`numpy` 用于 `day03` 的 memmap）。**不需要 librosa / torchaudio / transformers**，所以 Python 3.14 上也能全部跑通。

## common/ —— 12 天复用的训练循环

`common/trainer.py` 的唯一原则：**Trainer 不许知道你在训什么**。
它只管步数、学习率、bf16、梯度累积、裁剪、EMA、日志、checkpoint；
「什么是 loss」永远由你传进来的 `loss_fn` 决定。

```python
from common import TrainConfig, Trainer

def loss_fn(model, batch):          # 唯一需要你写的东西
    x, y = batch
    return F.cross_entropy(model(x), y)

Trainer(model, loss_fn, train_data, val_data,
        TrainConfig(max_steps=500, lr=3e-4)).fit()
```

`train_data` 可以是 DataLoader（自动无限循环），也可以是一个 `() -> batch`
的函数（LLM / 扩散那种随机采样的写法）。`loss_fn` 可以返回 `loss`，
也可以返回 `(loss, {"acc": ...})` 让额外指标进日志。

已经用同一个 Trainer 实跑验证过三个模态，一行都没改：
Day 4 的 GPT、Day 8 的 UNet+DDPM（带 EMA）、Day 13 的 AudioLM。

自检（10 秒，顺便确认你的 GPU 环境是好的）：

```bash
python common/trainer.py
```

带的东西：warmup+cosine 学习率、weight decay 只加在 2 维权重上、
bf16/fp16 自动切 GradScaler、梯度累积（已验证与大 batch 等价到 1e-8）、
梯度裁剪、EMA（`with ema.swapped(model): sample()`）、checkpoint 存取、
`print_shapes()` 一次前向打印所有中间 shape、
`smoke_test()`、以及内建的 `trainer.overfit_one_batch()`。

## 目录

```
ai-14days/
├─ check.py            题库统一入口
├─ common/
│  ├─ config.py        TrainConfig + 命令行覆盖
│  ├─ utils.py         set_seed / EMA / checkpoint / print_shapes / smoke_test
│  └─ trainer.py       通用训练循环（python common/trainer.py 可自检）
├─ drills/
│  ├─ _check.py        自测小工具（task / eq / shape_is / run）
│  └─ day01.py … day14.py
└─ solutions/
   └─ day01.py … day14.py
```
