"""校验讲义 .md 里写的输出是不是真的。

    python lessons/_verify_md.py lessons/day01a_basics.md

规则：一个 ```python 代码块，如果**紧跟着**一个无语言标记的 ``` 块（中间只有空行），
就把代码跑一遍，拿真实输出和那个块比对。不跟输出块的代码块跳过（那是示意代码）。
每个代码块在独立的命名空间里跑，所以每块都必须自包含。
"""
import contextlib
import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FENCE = re.compile(r"^```([a-zA-Z]*)[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)


def extract_pairs(src):
    """按出现顺序取出所有围栏块，再把「python 块 + 紧跟其后的无标记块」配成对。"""
    blocks = [(m.group(1), m.group(2), m.start(), m.end()) for m in FENCE.finditer(src)]
    pairs = []
    for i, (lang, body, _s, end) in enumerate(blocks):
        if lang != "python" or i + 1 >= len(blocks):
            continue
        nlang, nbody, nstart, _e = blocks[i + 1]
        if nlang:                      # 下一个块带语言标记，不是输出
            continue
        if src[end:nstart].strip():    # 两块之间夹了正文，说明这块没有配对输出
            continue
        pairs.append((body, nbody))
    return pairs, sum(1 for b in blocks if b[0] == "python")


def main(path):
    src = io.open(path, encoding="utf-8").read()
    pairs, total_py = extract_pairs(src)
    print(f"{path}: {total_py} 个 python 块，其中 {len(pairs)} 个带输出、需要校验\n")

    bad = 0
    for i, (code, want) in enumerate(pairs, 1):
        buf = io.StringIO()
        ns = {"__name__": "__main__"}
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)
        except Exception as e:
            print(f"[XX] 第 {i} 块 抛异常: {type(e).__name__}: {e}")
            print("     代码:\n" + "\n".join("       " + l for l in code.strip().splitlines()))
            bad += 1
            continue
        got = buf.getvalue()
        if got.rstrip("\n") != want.rstrip("\n"):
            print(f"[XX] 第 {i} 块 输出对不上")
            print("     讲义里写的:\n" + "\n".join("       " + l for l in want.rstrip().splitlines()))
            print("     实际跑出来:\n" + "\n".join("       " + l for l in got.rstrip().splitlines()))
            bad += 1
        else:
            first = code.strip().splitlines()[0][:52]
            print(f"[OK] 第 {i} 块  {first}")

    print()
    print(f"结果: {len(pairs) - bad} / {len(pairs)} 正确" + ("" if bad == 0 else f"，{bad} 处要修"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
