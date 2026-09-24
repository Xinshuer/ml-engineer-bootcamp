# 讲义

每份都是 **Markdown，代码和输出都写在里面，不用跑也能读完**。
所有输出都是真跑出来的（`_verify_md.py` 会把每段代码执行一遍比对，
`day01a/b/c` 三份共 57 个代码块全部核对通过）。

## Day 1

| 讲义 | 内容 | 配套练习 |
|---|---|---|
| [day01a_basics.md](day01a_basics.md) | 变量、缩进、列表、字典、切片；**函数与 `return` 的执行顺序**；**`for` 每一轮发生了什么**；赋值是贴标签不是复制 | `drills/day01.py` |
| [day01b_class.md](day01b_class.md) | **`self` 到底是谁**、`__init__`、`__call__`、继承与 `super().__init__()`、`@property` | `drills/day01.py` |
| [day01c_modelcode.md](day01c_modelcode.md) | dataclass、f-string、`*args/**kwargs`、import 与 `__main__`、pathlib、**报错速查表** | `drills/day01.py` |

想动手改着玩，用 [scratch.py](scratch.py)：把讲义里的代码块复制进去，改个数字再跑。

## 校验讲义

```bash
python lessons/_verify_md.py lessons/day01a_basics.md
```

它会把 md 里每个「python 块 + 紧跟的输出块」跑一遍做比对，防止讲义里的输出写错。
