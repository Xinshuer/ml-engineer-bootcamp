"""草稿本。把讲义里的代码复制到这里，改个数字再跑一次。

    python lessons/scratch.py

想清空就整个删掉重写，这个文件不参与任何检查。
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---- 在下面写你的实验 ----

xs = [10, 20, 30, 40, 50]
print(xs[1:3])
