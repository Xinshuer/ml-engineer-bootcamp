"""Map every drill task to the definitions it tests: which defs carry a TODO, which @task checks use which
names, and which solution symbols each check needs. Prints a table; used to design the content converter."""
import ast
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent / "classic"


def defs_with_todo(src, tree):
    out = {}
    lines = src.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            seg = "\n".join(lines[node.lineno - 1: node.end_lineno])
            for m in re.finditer(r"# --- TODO (\d+) ---", seg):
                out.setdefault(node.name, set()).add(int(m.group(1)))
            if "raise todo()" in seg and node.name not in out:
                out[node.name] = {"?"}
    return out


def names_used(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


for day in range(1, 15):
    f = ROOT / "drills" / f"day{day:02d}.py"
    src = f.read_text(encoding="utf-8")
    tree = ast.parse(src)
    todo = defs_with_todo(src, tree)
    top = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assigns = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    print(f"===== day{day:02d}: {len(todo)} defs with TODO; module-level names {sorted(assigns)}")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            d = node.decorator_list[0]
            if isinstance(d, ast.Call) and getattr(d.func, "id", "") == "task":
                tid = d.args[0].value
                title = d.args[1].value
                used = sorted((names_used(node) & (top | assigns)) - {node.name})
                own = [n for n, t in todo.items() if tid in t]
                print(f"  t{tid:02d} {title[:28]:28s} own={own} uses={used}")
