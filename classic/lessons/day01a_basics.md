# Day 1-A · Python 基础（零基础版）

> **怎么读这份讲义**
>
> 每一节都是：一段讲解 → 一块代码 → 这块代码跑出来的真实输出 → 一句要点。
> 输出已经帮你跑好了，**不用打开终端也能读完**。
>
> 想动手试，就把代码复制到 `lessons/scratch.py` 里跑，改个数字再跑一次。
>
> 这份材料只教「写 AI 模型代码会用到」的部分。用不上的东西列在 D 节最后，
> 这一周一次都碰不到，先别学。

---

## 目录

1. [print 和注释](#1-print-和注释)
2. [变量：给一个值起名字](#2-变量给一个值起名字)
3. [缩进就是语法](#3-缩进就是语法)
4. [列表 list](#4-列表-list)
5. [函数：def 和 return —— 执行顺序是重点](#5-函数def-和-return--执行顺序是重点)
6. [for 循环：每一轮到底发生了什么](#6-for-循环每一轮到底发生了什么)
7. [字典 dict](#7-字典-dict)
8. [元组 tuple](#8-元组-tuple)
9. [赋值是「贴标签」，不是「复制」](#9-赋值是贴标签不是复制)
10. [自检](#10-自检)

---

## 1. print 和注释

`print` 把东西显示出来。井号 `#` 后面的内容是注释，Python 完全忽略它，注释是写给人看的。

```python
print("你好")
print("一次打多个：", 1, "两", 3.0)
# 这一行是注释，不会执行
```

```
你好
一次打多个： 1 两 3.0
```

**要点**：Python 是**从上往下，一行一行执行**的。这听起来是废话，但后面讲函数和循环时，你唯一需要搞清楚的就是「现在执行到哪一行了」。

---

## 2. 变量：给一个值起名字

```python
x = 5
name = "小明"        # 字符串要用引号包起来
pi = 3.14
ok = True           # 布尔值只有 True / False，首字母大写

print(x, name, pi, ok)
print(type(x))
print(type(name))
print(type(3.14))
print(type([1, 2, 3]))
```

```
5 小明 3.14 True
<class 'int'>
<class 'str'>
<class 'float'>
<class 'list'>
```

**和 C / Java 不一样的三点**：

1. 不用声明类型。写 `x = 5` 就行，不用写 `int x = 5`。
2. 行尾不用分号。
3. 同一个名字可以先存整数、后存字符串，Python 不管。

### 算术

```python
print(7 / 3)      # 除法，永远得到小数
print(7 // 3)     # 整除，只要商
print(7 % 3)      # 取余数
print(7 ** 3)     # 乘方
print(-7 // 2)    # 整除是「向下取整」，不是「向零取整」
```

```
2.3333333333333335
2
1
343
-4
```

**要点**：`/` 和 `//` 的区别以后天天遇到。「768 维分给 12 个头，每个头多少维」必须写 `768 // 12`（得到整数 `64`），写 `768 / 12` 得到 `64.0` 这个小数，拿去描述张量形状会直接报错。

### 比较

```python
print(5 > 3)
print(5 == 5)         # 两个等号才是「相等吗」
print(5 != 3)         # 不等于
print(5 > 3 and 2 > 1)
print(not (5 > 3))
```

```
True
True
True
True
False
```

**要点**：一个等号 `=` 是**赋值**（把右边的值存进左边的名字）；两个等号 `==` 是**判断**（问「这两个一样吗」）。初学时最常打错的就是这个。

---

## 3. 缩进就是语法

别的语言用 `{ }` 表示「这一段属于 if / 属于函数」，Python 用**缩进**（行首的空格）。

```python
score = 85

if score >= 90:
    print("A")
elif score >= 80:            # elif = else if，可以有很多个
    print("B")
    print("这行也属于 elif，因为缩进一样")
else:
    print("C")

print("这行顶格，不管上面走哪一支都会执行")
```

```
B
这行也属于 elif，因为缩进一样
这行顶格，不管上面走哪一支都会执行
```

**三条规矩**：

1. 冒号 `:` 后面必须换行并缩进。
2. 同一个块里所有行缩进必须一模一样（统一用 4 个空格，**别用 Tab**）。
3. 缩进错了会报 `IndentationError`，改缩进就行。

---

## 4. 列表 list

列表是「一串按顺序排好的东西」。

```python
xs = [10, 20, 30, 40, 50]

print(len(xs))     # 有几个
print(xs[0])       # 第一个（下标从 0 开始，不是 1）
print(xs[-1])      # 倒数第一个
print(xs[1:3])     # 切片：下标 1、2，不含 3
print(xs[:3])      # 从头到下标 2
print(xs[2:])      # 从下标 2 到末尾
print(xs[::-1])    # 倒过来
```

```
5
10
50
[20, 30]
[10, 20, 30]
[30, 40, 50]
[50, 40, 30, 20, 10]
```

**「含头不含尾」很重要**。`xs[1:3]` 拿到的是 2 个元素不是 3 个，因为 `3` 是「停在哪」而不是「取到哪」。好处是 `xs[:k]` 和 `xs[k:]` 正好把列表切成互不重叠的两半。

### 这个规则马上就要用

语言模型的训练数据就是这么造的：

```python
data = list(range(10))
T = 4                      # 一次看 4 个
i = 3                      # 从第 3 个开始

x = data[i:i + T]
y = data[i + 1:i + 1 + T]  # 往右挪一格

print("data =", data)
print("x    =", x)
print("y    =", y)
```

```
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x    = [3, 4, 5, 6]
y    = [4, 5, 6, 7]
```

**要点**：`y` 里的每个元素，正好是 `x` 里对应位置的「下一个」。这就是「预测下一个词」的全部数学内容——两行切片。

> 这一行写错（比如忘了 `+1`），模型会学会「抄答案」：loss 掉到 0，生成全是垃圾，而且**不报任何错**。Day 2 的练习专门考它。

### 改列表

```python
xs = [10, 20, 30]
xs.append(40)      # 末尾加一个
print(xs)

xs[0] = 99         # 改某个位置
print(xs)

print(30 in xs)    # 判断在不在里面
```

```
[10, 20, 30, 40]
[99, 20, 30, 40]
True
```

---

## 5. 函数：def 和 return —— 执行顺序是重点

函数就是「一段起了名字的代码」。**定义它的时候不执行**，只有调用（写函数名加括号）时才执行。

### 先看清楚程序跳到哪里去了

```python
def double(n):
    print("  2. 进到函数里了，n =", n)
    result = n * 2
    print("  3. 算好了 result =", result)
    return result
    print("  这一行永远不会执行")

print("1. 调用之前")
r = double(5)
print("4. 回到外面，r =", r)
```

```
1. 调用之前
  2. 进到函数里了，n = 5
  3. 算好了 result = 10
4. 回到外面，r = 10
```

**执行路径**：

```
1. 打印「调用之前」
   ↓
   遇到 double(5)  →  跳进函数体
                     ↓
                     2. 打印
                     3. 打印
                     return  →  带着值 10 跳回来
   ↓
4. 打印，r 里存的就是那个 10
```

### return 的三件事

**(1) `return` 让函数立刻结束。** 后面的代码不会执行——上面那句「这一行永远不会执行」就没打出来。

**(2) `return` 的值必须接住才有用。**

```python
def double(n):
    return n * 2

double(5)              # 算完了，值扔掉了，什么都没发生
r = double(5)          # 值存进了 r
print(r)
print(double(5))       # 值直接给了 print
```

```
10
10
```

**(3) 没写 `return` 的函数，返回 `None`（表示「没有值」）。**

```python
def no_return(n):
    n * 2              # 算了，但没有 return

print(no_return(5))
```

```
None
```

> **这是初学者最常见的 bug 之一**：函数明明算对了，外面拿到的却是 `None`，
> 然后下一步报 `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`。
> 看到 `NoneType` 三个字，第一反应就是「哪个函数忘了 return」。

### 参数与默认值

```python
def power(base, exp=2):     # exp 有默认值，调用时可以不给
    return base ** exp

print(power(3))             # exp 用默认值 2
print(power(3, 3))          # 按位置传
print(power(3, exp=4))      # 按名字传，更清楚
```

```
9
27
81
```

### 一次返回多个值

```python
def min_and_max(nums):
    return min(nums), max(nums)      # 逗号隔开

both = min_and_max([3, 1, 4, 1, 5])
print(both)

lo, hi = min_and_max([3, 1, 4, 1, 5])   # 拆开接住，叫「解包」
print(lo, hi)
```

```
(1, 5)
1 5
```

**要点**：后面会大量看到这种写法：

```python
logits, loss = model(x, y)
q, k, v = qkv.split(C, dim=-1)
```

都是「函数返回了好几个东西，一次性拆开接住」。

### 函数里的变量出了函数就没了

```python
msg = "我在外面"

def try_change():
    msg = "我在里面"          # 这是函数内部一个新的变量
    print("函数内部：", msg)

try_change()
print("函数外部：", msg)
```

```
函数内部： 我在里面
函数外部： 我在外面
```

**要点**：函数内部赋值创建的是「局部变量」，函数一结束就消失，不影响外面。这是好事——你写函数时不用担心撞名字。

---

## 6. for 循环：每一轮到底发生了什么

`for` 的意思是：**把一堆东西一个一个拿出来，每拿一个，就把下面缩进的那段代码从头到尾执行一遍**。

```python
for fruit in ["苹果", "香蕉", "橘子"]:
    print("这一轮 fruit =", fruit)

print("循环结束")
```

```
这一轮 fruit = 苹果
这一轮 fruit = 香蕉
这一轮 fruit = 橘子
循环结束
```

循环体执行了 3 次，因为列表里有 3 个元素。每次开始前，Python 先把下一个元素赋给 `fruit`，再执行循环体。

### range(n)：生成 0, 1, 2, …, n-1

```python
for i in range(5):
    print("i =", i, " i*i =", i * i)

print(list(range(5)))
print(list(range(2, 6)))       # 从 2 到 5
print(list(range(0, 10, 3)))   # 步长 3
```

```
i = 0  i*i = 0
i = 1  i*i = 1
i = 2  i*i = 4
i = 3  i*i = 9
i = 4  i*i = 16
[0, 1, 2, 3, 4]
[2, 3, 4, 5]
[0, 3, 6, 9]
```

**要点**：`range(5)` 给的是 0~4，**不包含 5**。和切片一样是「含头不含尾」。Python 里凡是给范围，几乎都是这个规矩。

### 模式一：累加

```python
total = 0                      # 先准备一个「容器」
for n in [10, 20, 30, 40]:
    total = total + n          # 每一轮往里加
    print("拿到", n, "累计 total =", total)

print("最后 total =", total)
```

```
拿到 10 累计 total = 10
拿到 20 累计 total = 30
拿到 30 累计 total = 60
拿到 40 累计 total = 100
最后 total = 100
```

`total = total + n` 读作「把 total 加上 n 之后，再存回 total」，可以简写成 `total += n`。**训练循环里统计 loss 就是这个模式。**

### 模式二：边循环边造新列表

```python
squares = []                   # 先来一个空列表
for i in range(6):
    squares.append(i * i)      # 每一轮往里塞一个
print(squares)

# 上面 3 行有个等价的一行写法，叫「列表推导式」
print([i * i for i in range(6)])

# 还能带条件
print([i for i in range(10) if i % 3 == 0])
```

```
[0, 1, 4, 9, 16, 25]
[0, 1, 4, 9, 16, 25]
[0, 3, 6, 9]
```

**读推导式的方法**：先看中间的 `for`，再看最前面的表达式。

```
[  i * i    for i in range(6)  ]
   ↑每轮产出什么   ↑循环什么
```

### break 和 continue

```python
for i in range(10):
    if i == 3:
        continue        # 跳过这一轮剩下的，直接进下一轮
    if i == 6:
        break           # 整个循环立刻结束
    print("i =", i)
```

```
i = 0
i = 1
i = 2
i = 4
i = 5
```

### 两个常用搭档

```python
for i, ch in enumerate("abc"):       # 同时拿下标和元素
    print(i, ch)

names = ["lr", "batch"]
vals = [0.0003, 32]
for n, v in zip(names, vals):        # 两个列表配对
    print(n, "=", v)

xs = [3, 5, 4, 9]
for a, b in zip(xs, xs[1:]):         # 经典技巧：所有相邻对
    print(a, b, "差 =", b - a)
```

```
0 a
1 b
2 c
lr = 0.0003
batch = 32
3 5 差 = 2
5 4 差 = -1
4 9 差 = 5
```

**要点**：`zip` 在两边长度不一样时按短的停。所以 `zip(xs, xs[1:])` 正好是「所有相邻的两个一组」——Day 2 统计词对时要用。

---

## 7. 字典 dict

列表用数字下标（`xs[0]`），字典用你自己定的「键」（`d["lr"]`）。

```python
cfg = {"lr": 3e-4, "layers": 6}

print(cfg)
print(cfg["lr"])
print("lr" in cfg)

cfg["batch"] = 32                 # 加一个新的
print(cfg)

print(cfg.get("没有这个键"))       # 不存在时给 None，不报错
print(cfg.get("没有这个键", 0))    # 可以指定默认值
```

```
{'lr': 0.0003, 'layers': 6}
0.0003
True
{'lr': 0.0003, 'layers': 6, 'batch': 32}
None
0
```

**要点**：`cfg["不存在的键"]` 会直接报 `KeyError`；`cfg.get(...)` 不会。

统计次数时经常这么写——读作「取出 `c` 现在的计数（没有就当 0），加 1，再存回去」：

```python
counts = {}
for c in "abracadabra":
    counts[c] = counts.get(c, 0) + 1
print(counts)
```

```
{'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1}
```

### 遍历字典

```python
cfg = {"lr": 3e-4, "layers": 6}

for key in cfg:                  # 直接遍历，拿到的是键
    print("键", key)

for key, val in cfg.items():     # 键和值都要，用 .items()
    print(key, "=", val)

print({c: ord(c) for c in "abc"})   # 字典推导式
```

```
键 lr
键 layers
lr = 0.0003
layers = 6
{'a': 97, 'b': 98, 'c': 99}
```

**要点**：Day 2 你会用两行字典推导写完一个完整的分词器：

```python
stoi = {c: i for i, c in enumerate(chars)}    # 字符 -> 编号
itos = {i: c for i, c in enumerate(chars)}    # 编号 -> 字符
```

---

## 8. 元组 tuple

元组和列表几乎一样，唯一区别是**建好之后不能改**。

```python
t = (10, 20, 30)
print(t, t[0], len(t))

print((5,))     # 单个元素的元组，逗号不能省
print((5))      # 不加逗号就只是个括号，这是数字 5
```

```
(10, 20, 30) 10 3
(5,)
5
```

```python
t = (10, 20, 30)
try:
    t[0] = 99
except TypeError as e:
    print("报错了：", e)
```

```
报错了： 'tuple' object does not support item assignment
```

**为什么还要它？** 因为「不能改」在很多地方正好是想要的：

- 张量形状 `(2, 3, 4)` 天生就该是元组，不该被谁改掉
- 函数返回多个值，默认打包成元组（就是第 5 节那个 `(1, 5)`）

---

## 9. 赋值是「贴标签」，不是「复制」

> **这一节值得多花 10 分钟。它是后面六天里一大半诡异 bug 的根源。**

Python 里 `a = b` 从来不复制数据。它只是让 `a` 这个名字，指向 `b` 已经指向的那个东西——就像给同一个箱子贴第二张标签。

### 对列表（可以改的东西）

```python
a = [1, 2, 3]
b = a               # b 和 a 是同一个列表的两个名字
b.append(4)

print("a =", a)
print("a is b:", a is b)
```

```
a = [1, 2, 3, 4]
a is b: True
```

只动了 `b`，`a` 也变了。因为它们本来就是同一个列表。

### 想真的复制，必须说出来

```python
a = [1, 2, 3]
c = a.copy()        # 也可以写 list(a) 或 a[:]
c.append(4)

print("a =", a)
print("c =", c)
print("a is c:", a is c)
```

```
a = [1, 2, 3]
c = [1, 2, 3, 4]
a is c: False
```

### 对数字、字符串（不能改的东西）就没这个问题

```python
m = 5
n = m
n = n + 1           # 这是「让 n 指向一个新的数 6」，不是「把 5 改成 6」
print(m, n)
```

```
5 6
```

**分界线就是「这个类型能不能被就地改掉」**：

| | 类型 | 要不要小心 |
|---|---|---|
| 不能改 | `int` `float` `str` `tuple` | 不用 |
| 能改 | `list` `dict` `set`，以及后面的**张量和模型** | 要 |

### `==` 和 `is` 的区别

```python
p = [1, 2]
q = [1, 2]
print("p == q:", p == q)      # 值一样吗
print("p is q:", p is q)      # 是同一个东西吗
```

```
p == q: True
p is q: False
```

**要点**：Day 2 你会写这么一句，来确认「两层网络真的共用了同一份权重」：

```python
assert model.lm_head.weight is model.wte.weight
```

这里必须用 `is`。用 `==` 的话，两份数值碰巧相同的权重也会判成通过。

### 这条规则解释的第一个坑：可变默认参数

```python
def bad(x, acc=[]):     # 千万别这么写
    acc.append(x)
    return acc

print(bad(1))
print(bad(2))           # 你以为是 [2] 吧
print(bad(3))
```

```
[1]
[1, 2]
[1, 2, 3]
```

**为什么？** 因为 `acc=[]` 这个空列表，是在**定义函数的那一刻**建好的，**只建一次**。之后每次调用都在往同一个列表里塞东西。

正确写法固定是这个套路：

```python
def good(x, acc=None):
    if acc is None:     # 每次调用时现建一个新的
        acc = []
    acc.append(x)
    return acc

print(good(1))
print(good(2))
```

```
[1]
[2]
```

> `drills/day01.py` 第 12 题考的就是这个。

---

## 10. 自检

答不上来就回去翻对应小节：

1. 程序执行到 `r = double(5)` 这一行时，接下来跳到哪、再跳回哪？（第 5 节）
2. `return` 之后的代码会执行吗？没写 `return` 的函数返回什么？（第 5 节）
3. `for i in range(3)` 循环体执行几次？每次 `i` 是多少？（第 6 节）
4. `total = 0` 然后 `for ... total += n`，这个 `total` 起什么作用？（第 6 节）
5. `b = a` 之后改 `b`，`a` 会不会跟着变？分什么情况？（第 9 节）

---

**接着读**：[`day01b_class.md`](day01b_class.md) —— class 和 self
