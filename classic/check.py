"""统一入口。

    python check.py            跑全部 14 天，看总进度
    python check.py 04         只跑 Day 04
    python check.py 04 07      只跑第 7 题
    python check.py 04 --sol   用参考答案跑 Day 04（验证题目本身没问题）
    python check.py --all --sol  全部用参考答案跑一遍
"""
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run_day(day, extra_args, sol):
    env = dict(os.environ)
    if sol:
        env["DRILL_SOL"] = "1"
    cmd = [PY, os.path.join(ROOT, "drills", f"day{day}.py")] + extra_args
    return subprocess.run(cmd, env=env, cwd=ROOT)


def summarize():
    env = dict(os.environ)
    total_ok = total_all = 0
    rows = []
    for d in range(1, 15):
        day = f"{d:02d}"
        p = subprocess.run([PY, os.path.join(ROOT, "drills", f"day{day}.py")],
                           env=env, cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        m = re.search(r"通过 (\d+)/(\d+)", p.stdout or "")
        if not m:
            rows.append(f"  Day {day}   ?? 跑不起来（先单独跑 python drills/day{day}.py 看报错）")
            continue
        ok, n = int(m.group(1)), int(m.group(2))
        total_ok += ok
        total_all += n
        bar = "#" * round(20 * ok / n) + "." * (20 - round(20 * ok / n))
        rows.append(f"  Day {day}   [{bar}]  {ok:2d}/{n:2d}")
    print("=" * 46)
    print("  14 天题库总进度")
    print("=" * 46)
    print("\n".join(rows))
    print("-" * 46)
    print(f"  合计 {total_ok} / {total_all}")
    print("=" * 46)


if __name__ == "__main__":
    args = sys.argv[1:]
    sol = "--sol" in args
    args = [a for a in args if a != "--sol"]

    if not args or args[0] == "--all":
        if sol:
            for d in range(1, 15):
                run_day(f"{d:02d}", [], True)
        else:
            summarize()
    else:
        day = args[0].zfill(2)
        raise SystemExit(run_day(day, args[1:], sol).returncode)
