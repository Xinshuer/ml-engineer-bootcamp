# Day 1-C · 模型代码里的固定套路（零基础版）

> 前置：[`day01a_basics.md`](day01a_basics.md)、[`day01b_class.md`](day01b_class.md)
>
> 这一节把你打开任何一个开源训练项目都会看到的那几样东西讲完。
> 学完这份，你读 nanoGPT / diffusers 的代码就不会再被语法本身挡住。

---

## 目录

1. [dataclass：所有训练脚本的入口](#1-dataclass所有训练脚本的入口)
2. [f-string：训练日志天天写](#2-f-string训练日志天天写)
3. [`*args` / `**kwargs` 和装饰器](#3-args--kwargs-和装饰器)
4. [import、模块、`if __name__ == "__main__"`](#4-import模块if-__name__--__main__)
5. [pathlib：checkpoint 路径管理](#5-pathlibcheckpoint-路径管理)
6. [数值上的两个坑](#6-数值上的两个坑)
7. [报错速查表](#7-报错速查表)
8. [刻意不学的东西](#8-刻意不学的东西)

---

## 1. dataclass：所有训练脚本的入口

你打开任何一个开源训练项目，入口都是「一个配置对象 + 一个 main 函数」。配置对象用 `dataclass` 写最省事：**只写字段名和默认值**，`__init__`、打印格式、比较都自动生成。

```python
from dataclasses import dataclass, field, asdict, replace

@dataclass
class Cfg:
    lr: float = 3e-4                            # 冒号后面是「类型注解」
    n_layer: int = 6
    device: str = "cuda"
    betas: tuple = (0.9, 0.95)                  # 元组不可变，可以直接当默认值
    tags: list = field(default_factory=list)    # 列表可变，必须用 field

    def scaled_lr(self, k):                     # dataclass 里照样能写方法
        return self.lr * k

c = Cfg()
print(c)
print(c.lr, c.n_layer)
print(Cfg(lr=1e-3).scaled_lr(2))
```

```
Cfg(lr=0.0003, n_layer=6, device='cuda', betas=(0.9, 0.95), tags=[])
0.0003 6
0.002
```

### 为什么可变字段必须写 `field(default_factory=list)`

这就是 [day01a 第 9 节](day01a_basics.md#9-赋值是贴标签不是复制)那个「可变默认参数」的坑。`dataclass` 直接帮你拦住了：

```python
from dataclasses import dataclass

try:
    @dataclass
    class Bad:
        tags: list = []          # 直接写 [] 会报错
except ValueError as e:
    print("报错了：", e)
```

```
报错了： mutable default <class 'list'> for field tags is not allowed: use default_factory
```

用 `field(default_factory=list)` 之后，每个实例都会**现建一个新列表**，互不干扰：

```python
from dataclasses import dataclass, field

@dataclass
class Cfg:
    tags: list = field(default_factory=list)

a = Cfg()
b = Cfg()
a.tags.append("实验1")
print("a.tags =", a.tags)
print("b.tags =", b.tags)     # 没被污染
```

```
a.tags = ['实验1']
b.tags = []
```

### 两个实用工具

```python
from dataclasses import dataclass, asdict, replace

@dataclass
class Cfg:
    lr: float = 3e-4
    n_layer: int = 6

c = Cfg()
print(asdict(c))                 # 转成字典，存 checkpoint 时用
print(replace(c, lr=1e-2))       # 复制一份、只改一个字段，做消融实验时用
print(c)                         # 原来那个没被动
```

```
{'lr': 0.0003, 'n_layer': 6}
Cfg(lr=0.01, n_layer=6)
Cfg(lr=0.0003, n_layer=6)
```

### 类型注解不检查

```python
from dataclasses import dataclass

@dataclass
class Cfg:
    lr: float = 3e-4

print(Cfg(lr="这不是数字").lr)     # Python 根本不管
```

```
这不是数字
```

**要点**：别指望注解帮你抓 bug，它只是给人和编辑器看的。写模型代码时，**最有用的「注解」其实是张量形状的注释**：

```python
x = self.wte(idx)        # (B, T, C)
q = q.transpose(1, 2)    # (B, nh, T, hs)
```

这个习惯从 Day 1 下午开始养，比任何类型系统都管用。

---

## 2. f-string：训练日志天天写

字符串前面加个 `f`，就能在 `{}` 里直接塞变量和表达式。

```python
step, loss, lr, dt = 1200, 1.48321, 3e-4, 0.12534

print(f"{step}")
print(f"{step:5d}")            # 右对齐，宽度 5
print(f"{step:<5d}|")          # 左对齐
print(f"{loss:.4f}")           # 保留 4 位小数
print(f"{lr:.2e}")             # 科学计数法
print(f"{dt * 1000:.1f}ms")    # 可以先算再格式化
print(f"{loss=}")              # 调试神器：自动带上变量名
```

```
1200
 1200
1200 |
1.4832
3.00e-04
125.3ms
loss=1.48321
```

拼起来就是训练日志：

```python
step, loss, lr, dt = 1200, 1.48321, 3e-4, 0.12534
print(f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | {dt * 1000:.1f}ms")
```

```
step  1200 | loss 1.4832 | lr 3.00e-04 | 125.3ms
```

**要点**：对齐很重要。数字不对齐，loss 的下降趋势你一眼看不出来。

---

## 3. `*args` / `**kwargs` 和装饰器

`*args` 把「所有按位置传进来的参数」收成一个元组，`**kwargs` 把「所有按名字传进来的」收成一个字典。

```python
def show_all(*args, **kwargs):
    print("args   =", args)
    print("kwargs =", kwargs)

show_all(1, 2, 3, lr=0.01, name="试验")
```

```
args   = (1, 2, 3)
kwargs = {'lr': 0.01, 'name': '试验'}
```

反过来，加星号可以把元组 / 字典**拆开**传出去：

```python
def power(base, exp):
    return base ** exp

args = (2, 10)
kw = {"base": 3, "exp": 4}
print(power(*args))     # 等价于 power(2, 10)
print(power(**kw))      # 等价于 power(base=3, exp=4)
```

```
1024
81
```

### 装饰器：`@deco` 就是语法糖

```
@deco
def f(...): ...          等价于        f = deco(f)
```

自己写装饰器的场合很少，但要能看懂它怎么把参数原样转发出去：

```python
import functools

def logged(fn):
    @functools.wraps(fn)              # 保住原函数的名字
    def wrapper(*args, **kwargs):     # 收下所有参数
        print(f"  [调用] {fn.__name__} args={args} kwargs={kwargs}")
        return fn(*args, **kwargs)    # 原样转发
    return wrapper

@logged
def add(a, b, scale=1):
    return (a + b) * scale

print(add(1, 2))
print(add(1, 2, scale=10))
print(add.__name__)
```

```
  [调用] add args=(1, 2) kwargs={}
3
  [调用] add args=(1, 2) kwargs={'scale': 10}
30
add
```

**要点**：这一周你只需要**认得** `@dataclass`、`@property`、`@torch.no_grad()` 这三个，不需要自己写装饰器。

---

## 4. import、模块、`if __name__ == "__main__"`

一个 `.py` 文件就是一个**模块**。`import` 它的时候，这个文件会**从头到尾执行一遍**，然后把里面定义的名字挂到模块对象上。

```python
import math                    # 整个模块
import torch.nn as nn          # 起别名，最常见
from math import sqrt, pi      # 从模块里取几个名字出来

print(math.sqrt(16))
print(sqrt(16))
print(f"{pi:.4f}")
```

```
4.0
4.0
3.1416
```

### `__name__` 是什么

Python 给每个模块一个内置变量 `__name__`：

| 情况 | `__name__` 的值 |
|---|---|
| 这个文件被**直接执行**（`python train.py`） | `"__main__"` |
| 这个文件被**import 进来** | 模块名，比如 `"train"` |

```python
print(__name__)
```

```
__main__
```

所以标准入口写法是：

```python
def main():
    ...训练代码...

if __name__ == "__main__":
    main()
```

**为什么必须这么写？** 两个理由：

1. 别人 `import` 你的文件时（比如只想复用你的模型类），不该顺带把训练跑起来。
2. **Windows 上更要命**：`DataLoader` 开多进程时，子进程会重新 import 主脚本。没有这个保护，就会**无限递归启动新进程**，直接卡死机器。

### 你一定会遇到的 import 报错

```
ImportError: attempted relative import with no known parent package
```

原因：你直接 `python model.py` 跑了一个用了**相对导入**（`from .xxx import yyy`，注意那个点）的文件。相对导入需要这个文件被当作「某个包的一部分」导入，而不是当脚本执行。

本课统一用绝对导入 + 在项目根目录跑，避开这个问题：

```
cd G:/个人项目/ai-14days
python drills/day04.py            # 文件里写 from common.trainer import Trainer
```

---

## 5. pathlib：checkpoint 路径管理

`pathlib` 把 `/` 重载成了「拼路径」，比手动拼字符串安全得多（不用操心 Windows 的反斜杠）。

```python
from pathlib import Path

p = Path("ckpt") / "gpt-tiny" / "ckpt_000500.pt"

print(p.name)       # 文件名
print(p.stem)       # 去掉扩展名
print(p.suffix)     # 扩展名
print(p.parent.name)
print(p.exists())
```

```
ckpt_000500.pt
ckpt_000500
.pt
gpt-tiny
False
```

最常用的两句：

```python
d.mkdir(parents=True, exist_ok=True)   # 一路把目录建出来，已存在也不报错
p = d / f"ckpt_{step:06d}.pt"          # 步数补零
```

**为什么补零**：`ckpt_9.pt` 和 `ckpt_10.pt` 按字符串排序，`10` 会排在 `9` 前面。补零成 `ckpt_000009.pt` / `ckpt_000010.pt` 才对。

```python
steps = [9, 10, 100]
print(sorted(f"ckpt_{s}.pt" for s in steps))        # 错的
print(sorted(f"ckpt_{s:06d}.pt" for s in steps))    # 对的
```

```
['ckpt_10.pt', 'ckpt_100.pt', 'ckpt_9.pt']
['ckpt_000009.pt', 'ckpt_000010.pt', 'ckpt_000100.pt']
```

---

## 6. 数值上的两个坑

### 坑一：浮点数不能用 `==` 比较

```python
print(0.1 + 0.2)
print(0.1 + 0.2 == 0.3)
print(abs(0.1 + 0.2 - 0.3) < 1e-9)     # 正确的比较方式：带容差
```

```
0.30000000000000004
False
True
```

**要点**：本课 `drills/` 里所有数值比较都走 `torch.allclose` 或带容差的 `eq(...)`，原因就是这个。

### 坑二：`/` 和 `//`

```python
n_embd, n_head = 768, 12

print(n_embd / n_head)       # 得到小数
print(n_embd // n_head)      # 得到整数

try:
    "x" * (n_embd / n_head)  # 小数不能当次数用，会报错
except TypeError as e:
    print("报错了：", e)
```

```
64.0
64
报错了： can't multiply sequence by non-int of type 'float'
```

**要点**：凡是算形状、算头数、算 patch 数，一律用 `//`。用 `/` 得到的 `64.0` 拿去 `view()` 会直接报错。

---

## 7. 报错速查表

零基础阶段 90% 的报错就这几种。看到报错先对着这张表找，别慌。

| 报错 | 意思 | 通常是因为 |
|---|---|---|
| `IndentationError` | 缩进不对 | 缩进不统一，或者混用了 Tab 和空格 |
| `SyntaxError` | 语法写错了 | 少了冒号 `:`、少了括号、中文标点混进来了 |
| `NameError: name 'x' is not defined` | 这个名字不存在 | 拼错了；或者变量是别的函数里的局部变量 |
| `TypeError: ... 'NoneType' ...` | 拿 `None` 去做运算了 | **某个函数忘了写 `return`** |
| `AttributeError: 'X' object has no attribute 'y'` | 对象上没有这个东西 | 忘了在 `__init__` 里 `self.y = ...`；或忘了 `super().__init__()` |
| `KeyError: 'k'` | 字典里没这个键 | 拼错了；或者该用 `.get(k, 默认值)` |
| `IndexError: list index out of range` | 下标越界 | 下标从 0 开始，最大是 `len(xs) - 1` |
| `ImportError: attempted relative import...` | 相对导入用错了 | 在项目根目录跑，改用绝对导入 |
| `ValueError: not enough values to unpack` | 解包时左右个数对不上 | `a, b = f()` 但 `f` 只返回了一个值 |

**读报错的方法**：报错信息**从下往上看**。最后一行是「什么错」，倒数第二行往上是「在哪一行出的错」。中间那一大堆是调用链，先不管。

---

## 8. 刻意不学的东西

下面这些是 Python 的重要特性，但写 AI 模型代码这一周**一次都用不上**。现在学等于浪费时间，需要的时候再回来查。

| 不学 | 理由 |
|---|---|
| `asyncio` / `async`-`await` | 没有并发 IO 的场景 |
| 元类 metaclass | 你不会写框架 |
| 多继承与 MRO | `nn.Module` 单继承就够 |
| 带参数的装饰器（装饰器工厂） | 会用 `@torch.no_grad()` 即可 |
| `typing` 的高级用法 | `Protocol` / `TypeVar` / `Generic` 全都不需要 |
| 抽象基类 `ABC` | `nn.Module` 已经替你做了 |
| 自定义上下文管理器 | 会用 `with torch.no_grad():` 就行 |
| 生成器与 `yield` | `DataLoader` 已经封好了 |
| 正则表达式 | 只有 Day 2 的 BPE 可能碰一点，到时再说 |

**一句话原则**：这一周你写的每一行 Python，都应该是在描述**张量怎么流动**。凡是不服务于这个目标的语言特性，都可以先放着。

---

## 读完之后

三份讲义读完了，接下来是今天的第二个动作——做题：

```
cd G:/个人项目/ai-14days
python drills/day01.py        # 先看一眼 12 道题长什么样
python drills/day01.py 05     # 只跑第 5 题
```

把 `raise todo()` 换成你的实现，跑到全 `[OK]` 为止。卡住超过 5 分钟再看 `solutions/day01.py`。

做完 `day01.py`，今天下午的内容是 `drills/day02.py`（PyTorch 张量），到时我再给对应的讲义。
