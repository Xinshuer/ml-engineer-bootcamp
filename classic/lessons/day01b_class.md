# Day 1-B · class 和 self（零基础版）

> 前置：先读完 [`day01a_basics.md`](day01a_basics.md)（变量、函数、`return`、`for`）。
>
> 这一节只有一个目标：**让你彻底不怕 `self`**。
> 读完之后，你看 PyTorch 的模型代码不会再有「这个 self 是从哪冒出来的」这种感觉。

---

## 目录

1. [为什么需要 class](#1-为什么需要-class)
2. [class 是图纸，实例是照图纸造出来的东西](#2-class-是图纸实例是照图纸造出来的东西)
3. [self 到底是什么 —— 本节核心](#3-self-到底是什么--本节核心)
4. [`__init__`：造对象时自动被调用的那段](#4-__init__造对象时自动被调用的那段)
5. [`__call__`：让对象能像函数一样被调用](#5-__call__让对象能像函数一样被调用)
6. [继承：拿别人写好的，改一点点](#6-继承拿别人写好的改一点点)
7. [`@property`](#7-property)
8. [拼起来：真实的模型骨架长什么样](#8-拼起来真实的模型骨架长什么样)
9. [自检](#9-自检)

---

## 1. 为什么需要 class

假设你要做一个计数器：记住当前数到几，可以往上加。不用 class 的话：

```python
count = 0

def add(n):
    global count          # 要改函数外面的变量，得写 global，很别扭
    count = count + n
    return count

print(add(5))
print(add(3))
```

```
5
8
```

能用。但现在问题来了：**我要两个互不干扰的计数器怎么办？**

```
count_a = 0
count_b = 0
def add_a(n): ...
def add_b(n): ...
```

三个计数器写三份，十个写十份。显然不对。

**问题的本质**：「一份数据」和「操作这份数据的函数」应该被打包在一起，然后可以按需要造出很多份。这个打包，就叫 class。

---

## 2. class 是图纸，实例是照图纸造出来的东西

```python
class Counter:                    # 定义一张图纸，名字叫 Counter
    def __init__(self, start):    # 造东西时自动执行的那段代码
        self.n = start            # 给造出来的这个东西，挂一个属性 n

    def add(self, k):             # 图纸上的一个操作
        self.n = self.n + k
        return self.n

a = Counter(0)                    # 按图纸造第一个，起始值 0
b = Counter(100)                  # 按图纸造第二个，起始值 100

print(a.n, b.n)
print(a.add(5))
print(a.add(5))
print(b.add(1))
print(a.n, b.n)
```

```
0 100
5
10
101
10 101
```

**要点**：两个计数器完全独立。`class` 只是图纸，**定义它的时候什么都没造出来，`__init__` 也没执行**；写 `Counter(0)` 才真的造一个。

术语对照：

| 说法 | 指的是 |
|---|---|
| 类（class） | `Counter` 这张图纸 |
| 实例（instance）/ 对象 | `a`、`b` 这些照图纸造出来的东西 |
| 属性（attribute） | `a.n` —— 挂在实例上的数据 |
| 方法（method） | `a.add(...)` —— 挂在实例上的函数 |

---

## 3. self 到底是什么 —— 本节核心

很多人卡在这：「`add(self, k)` 明明有两个参数，我怎么只传了一个？」

**答案：`self` 就是点号左边那个对象，Python 自动帮你传进去了。**

```
a.add(5)                你写的
     ↓
Counter.add(a, 5)       Python 实际执行的
```

不信？下面两种写法结果完全一样：

```python
class Counter:
    def __init__(self, start):
        self.n = start
    def add(self, k):
        self.n = self.n + k
        return self.n

c1 = Counter(0)
c2 = Counter(0)

print(c1.add(7))              # 正常写法
print(Counter.add(c2, 7))     # 手动把对象当第一个参数传进去，完全等价
```

```
7
7
```

所以：

- `self` **不是关键字**，只是个名字。写成 `me`、`this` 也能跑，但全世界都写 `self`，你也必须写 `self`。
- 定义方法时必须把它写在第一个参数位；调用时不用传。
- `self` 指向「当前正在被操作的那个实例」。`a.add(5)` 时 self 就是 `a`；`b.add(5)` 时 self 就是 `b`。
- **同一份代码，作用在不同的对象上** —— 这就是 class 的全部魔法。

### `self.x` 和 `x` 是两个完全不同的东西

```python
class Demo:
    def __init__(self):
        self.saved = "挂在对象上"
        temp = "只是个局部变量"
        print("[__init__] temp =", temp)

    def look(self):
        print("[look] self.saved =", self.saved)

d = Demo()
d.look()
print("外面也读得到：", d.saved)
```

```
[__init__] temp = 只是个局部变量
[look] self.saved = 挂在对象上
外面也读得到： 挂在对象上
```

而没挂到 `self` 上的东西，出了那个函数就没了：

```python
class Demo2:
    def __init__(self):
        temp = "只是个局部变量"

    def look(self):
        try:
            print(temp)
        except NameError as e:
            print("报错了：", e)

Demo2().look()
```

```
报错了： name 'temp' is not defined
```

**一句话规则：想让数据活过这个函数，就挂到 `self` 上。**

这解释了模型代码里为什么每一层都要写 `self`：

```python
class Block:
    def __init__(self, dim):
        self.ln = nn.LayerNorm(dim)    # 挂上去，forward 里才用得到
        temp = nn.LayerNorm(dim)       # 这么写等于白建，出了 __init__ 就没了

    def forward(self, x):
        return self.ln(x)              # 只有 self.ln 拿得到
```

---

## 4. `__init__`：造对象时自动被调用的那段

```python
class Talker:
    def __init__(self, name):
        print("  [__init__] 我被自动调用了，name =", name)
        self.name = name

print("造对象之前")
t = Talker("小明")
print("造完了，t.name =", t.name)
```

```
造对象之前
  [__init__] 我被自动调用了，name = 小明
造完了，t.name = 小明
```

**完整过程是三步**：

1. Python 先造一个空壳对象
2. 自动调用 `__init__(空壳, "小明")`，让你往壳里填东西
3. 把填好的对象交给你（存进 `t`）

两个细节：

- `__init__` **没有 `return`**。它的任务是「布置好 self」，不是「返回对象」。在 `__init__` 里 `return` 一个值会直接报错。
- 两边的双下划线读作 **dunder**（double underscore）。带 dunder 的方法都是「Python 在特定时机自动帮你调用」的钩子。

---

## 5. `__call__`：让对象能像函数一样被调用

```python
class Doubler:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):        # 定义了它，对象就能加括号调用
        return x * self.factor

dbl = Doubler(2)
print(dbl(21))
print(dbl.__call__(21))           # 上面那行实际走的就是这条路
```

```
42
42
```

**这个东西直接解释了 PyTorch 里最常见的一个疑问：**

```python
model = GPT(...)
y = model(x)          # model 是个对象，为什么能加括号？
```

因为 `nn.Module` 定义了 `__call__`。而它的 `__call__` 内部会去调用**你写的 `forward`**，顺便处理一些注册好的钩子。

```
你写：   def forward(self, x): ...
你调：   model(x)          ← 不要写 model.forward(x)
```

写 `model.forward(x)` 会绕过那些钩子，是常见的错误写法。

---

## 6. 继承：拿别人写好的，改一点点

继承的意思是：「我这个新图纸，先把旧图纸的所有东西照抄一份，然后我再加几样、改几样」。写法是在类名后面加括号写父类。

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return self.name + " 发出了声音"

    def intro(self):
        return "我是 " + self.name + "。" + self.speak()

class Dog(Animal):                # Dog 继承 Animal
    def speak(self):              # 覆盖掉父类的 speak
        return self.name + " 说：汪"

print(Animal("动物").speak())
print(Dog("旺财").speak())
print(Dog("旺财").intro())
```

```
动物 发出了声音
旺财 说：汪
我是 旺财。旺财 说：汪
```

**注意最后一行**：`intro` 这个方法 `Dog` 根本没写，是从父类继承来的。但它里面调用的 `self.speak()`，找到的是**子类**的 `speak`——因为 `self` 是那个 Dog 对象，查方法时先在 `Dog` 里找。

**PyTorch 就是靠这个工作的**：`nn.Module` 写好了 `__call__`、参数注册、`.to(device)`、`.train()` / `.eval()` 等一大堆东西，你继承它，只写 `forward`。

### `super().__init__()` —— 为什么每个模型的第一行都是它

```python
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        return self.name + " 发出了声音"

class BadDog(Animal):
    def __init__(self, name, color):
        self.color = color            # 忘了调父类的 __init__

class GoodDog(Animal):
    def __init__(self, name, color):
        super().__init__(name)        # 先让父类把它那部分布置好
        self.color = color            # 再加自己的

try:
    BadDog("小黑", "黑").speak()
except AttributeError as e:
    print("BadDog 报错：", e)

g = GoodDog("小黑", "黑")
print(g.speak(), "|", g.color)
```

```
BadDog 报错： 'BadDog' object has no attribute 'name'
小黑 发出了声音 | 黑
```

**原因很直白**：`self.name` 是在 `Animal.__init__` 里挂上去的。子类自己写了 `__init__`，就把父类那段**盖掉了**，不主动调就永远不执行。

所以每一个 `nn.Module` 子类的 `__init__` 第一行永远是：

```python
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()          # 少了这行，PyTorch 的参数注册机制没建起来
        self.ln = nn.LayerNorm(dim)
```

忘了写会报 `cannot assign module before Module.__init__() call`。看到这个报错，回来加这一行就行。

---

## 7. `@property`

```python
class Progress:
    def __init__(self, step, total):
        self.step = step
        self.total = total

    @property                       # 加了这一行
    def ratio(self):
        return self.step / self.total

    def ratio_method(self):         # 没加的对照组
        return self.step / self.total

p = Progress(25, 100)
print(p.ratio)                      # 没有括号
print(p.ratio_method())             # 有括号
```

```
0.25
0.25
```

`@property` 唯一的作用是「调用时不用写括号」，让算出来的东西用起来像个普通属性。你几乎不需要自己写它，但要认得出来，否则看到 `p.ratio` 会以为它是个存好的变量。

`@` 开头的这一行叫**装饰器**。这一周只需要认识三个：

| 装饰器 | 作用 |
|---|---|
| `@property` | 方法伪装成属性 |
| `@dataclass` | 把类变成配置容器（下一份讲义讲） |
| `@torch.no_grad()` | 「这个函数里不要记录求导用的信息」（推理 / 采样时用） |

自己写装饰器的场合极少，认得会用就够了。

---

## 8. 拼起来：真实的模型骨架长什么样

下面这段是 Day 2 你会亲手写的东西的简化版。**现在不用懂它在算什么**，只要能指出「哪个是 self、哪个是继承、哪个是 `__init__`、`forward` 是被谁调用的」就够了。

```python
import torch.nn as nn

class Block(nn.Module):                    # 继承 nn.Module
    def __init__(self, dim):
        super().__init__()                 # 必须，第一行
        self.ln = nn.LayerNorm(dim)        # 挂到 self 上，forward 才用得到
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):                  # 你只写这个
        return x + self.fc(self.ln(x))     # 一个残差连接

block = Block(384)        # 造对象 -> 自动执行 __init__
y = block(x)              # 加括号 -> nn.Module 的 __call__ -> 你的 forward
```

逐字对应这一节讲过的：

| 这一行 | 对应第几节 |
|---|---|
| `class Block(nn.Module)` | 第 6 节 · 继承 |
| `super().__init__()` | 第 6 节 · 先让父类布置好 |
| `self.ln = ...` | 第 3 节 · 挂到对象上才活得过 `__init__` |
| `def forward(self, x)` | 第 3 节 · self 是「点号左边那个对象」 |
| `block(x)` | 第 5 节 · `__call__` 让对象能被调用 |

---

## 9. 自检

答不上来就回去翻对应小节：

1. `a.add(5)` 时，`self` 是谁？Python 实际执行的是哪一句？（第 3 节）
2. `__init__` 是谁调用的？什么时候？（第 4 节）
3. `self.x = 1` 和 `x = 1` 写在方法里，有什么区别？（第 3 节）
4. `model(x)` 为什么能加括号？它最终调到了你写的哪个方法？（第 5 节）
5. 为什么每个 `nn.Module` 的 `__init__` 第一行都是 `super().__init__()`？（第 6 节）

---

**接着读**：[`day01c_modelcode.md`](day01c_modelcode.md) —— 模型代码里的固定套路
