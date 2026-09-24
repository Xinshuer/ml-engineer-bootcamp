"""One-off converter: classic/drills + classic/solutions (the terminal version of the course) ->
content/dayN.txt drafts (Chinese) + content/ref/dayN.py (reference implementations used as hidden setup).

The drafts carry everything that can be extracted mechanically (starter, solution, checks, targets, ref).
Background notes, knowledge cards, hints and the English version are written afterwards by hand.

    python tools/convert_classic.py            # writes content/ref/*.py and content/dayN.txt (refuses to overwrite)
    python tools/convert_classic.py --force    # overwrite existing drafts
"""
import ast
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
CLASSIC = ROOT / "classic"
CONTENT = ROOT / "content"
REF = CONTENT / "ref"
FORCE = "--force" in sys.argv
PARTS = {1: "地基", 2: "地基", 3: "大语言模型", 4: "大语言模型", 5: "大语言模型", 6: "大语言模型", 7: "图像生成", 8: "图像生成",
         9: "图像生成", 10: "图像生成", 11: "音频", 12: "音频", 13: "音频", 14: "元技能"}
REPORT = []


def seg(lines, a, b):
    return "\n".join(lines[a - 1:b])


def node_names(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    return set()


def start_line(node):
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def is_task(node):
    return (isinstance(node, ast.FunctionDef) and node.decorator_list and isinstance(node.decorator_list[0], ast.Call)
            and getattr(node.decorator_list[0].func, "id", "") == "task")


def comment_block_above(lines, first):
    """The comment lines (and the `# 01 title` banner) right above a node, as plain text."""
    out = []
    i = first - 2
    while i >= 0 and (lines[i].startswith("#") or not lines[i].strip()):
        out.append(lines[i])
        i -= 1
    out.reverse()
    text = []
    for ln in out:
        s = ln.lstrip("#").rstrip()
        if re.match(r"^\s*=+\s*$", s) or re.match(r"^\s*-+ TODO", s):
            continue
        text.append(s[1:] if s.startswith(" ") else s)
    return "\n".join(text).strip("\n")


def convert(day):
    dsrc = (CLASSIC / "drills" / f"day{day:02d}.py").read_text(encoding="utf-8")
    ssrc = (CLASSIC / "solutions" / f"day{day:02d}.py").read_text(encoding="utf-8")
    dl, sl = dsrc.split("\n"), ssrc.split("\n")
    dtree, stree = ast.parse(dsrc), ast.parse(ssrc)
    top = [n for n in dtree.body if node_names(n) and not is_task(n)]
    sol_nodes = {}
    for n in stree.body:
        for name in node_names(n):
            sol_nodes[name] = n

    # TODO markers -> task number -> node names. Column-0 markers belong to the next definition,
    # indented ones to the enclosing (previous) one.
    todo_of = {}
    for i, ln in enumerate(dl, 1):
        m = re.search(r"# --- TODO (\d+)\w? ---", ln)
        if not m:
            continue
        tid = int(m.group(1))
        indented = ln[: len(ln) - len(ln.lstrip())] != ""
        owner = None
        if indented:
            prev = [n for n in top if start_line(n) <= i]
            owner = prev[-1] if prev else None
        else:
            nxt = [n for n in top if start_line(n) > i]
            owner = nxt[0] if nxt else None
        if owner is not None:
            todo_of.setdefault(tid, [])
            for name in node_names(owner):
                if name not in todo_of[tid]:
                    todo_of[tid].append(name)

    doc = ast.get_docstring(dtree) or ""
    title_line = doc.split("\n")[0]
    title = re.sub(r"^Day \d+ · ", "", re.sub(r"\s*—+\s*\d+ 题\s*$", "", title_line)).strip()
    items = []            # dicts: tid, title, targets, type, checks[str], prompt
    names_defined = {n2: n for n in top for n2 in node_names(n)}
    tasks = [n for n in dtree.body if is_task(n)]
    for t in tasks:
        tid = t.decorator_list[0].args[0].value
        ttitle = t.decorator_list[0].args[1].value
        used = {n.id for n in ast.walk(t) if isinstance(n, ast.Name)} & set(names_defined)
        check_src = seg(dl, start_line(t), t.end_lineno)
        check_src = re.sub(r"^@task\(\s*\d+\s*,\s*", "@check(", check_src, count=1)
        targets = list(todo_of.get(tid, []))
        kind = "code"
        if ttitle.startswith("找 bug"):
            kind = "fix"
            targets = [u for u in used if u in sol_nodes and seg(dl, start_line(names_defined[u]), names_defined[u].end_lineno).strip()
                       != seg(sl, start_line(sol_nodes[u]), sol_nodes[u].end_lineno).strip() and not todo_owner(todo_of, u)]
        if not targets:
            consts = [u for u in used if isinstance(names_defined[u], ast.Assign) and u in sol_nodes]
            if consts:
                sol_val = seg(sl, sol_nodes[consts[0]].lineno, sol_nodes[consts[0]].end_lineno)
                if re.search(r"=\s*True\s*$", sol_val.strip()):
                    kind = "self"
                targets = consts
        if not targets:
            # a check on something an earlier task defines: attach it to the latest item that owns one of the names
            owners = [it for it in items if set(it["targets"]) & used]
            if owners:
                owners[-1]["checks"].append(check_src)
                owners[-1]["merged"].append(f"t{tid:02d} {ttitle}")
                continue
            kind = "demo"
            REPORT.append(f"day {day} t{tid:02d} {ttitle}: no learner code (demo) - needs manual handling")
        first = min(start_line(names_defined[n]) for n in targets) if targets else start_line(t)
        items.append({"tid": tid, "title": ttitle, "targets": targets, "type": kind, "checks": [check_src], "merged": [],
                      "prompt": comment_block_above(dl, first)})

    ref = ssrc
    extra_ref = []
    for m in re.finditer(r"^from solutions\.day(\d+) import .*$", ssrc + "\n" + dsrc, flags=re.M):
        if f"day{int(m.group(1))}" not in extra_ref:
            extra_ref.append(f"day{int(m.group(1))}")
    ref = re.sub(r"^from solutions\.day\d+ import .*\n", "", ref, flags=re.M)
    ref = re.sub(r"^sys\.path\.insert.*\n", "", ref, flags=re.M)
    # the checks were written against the drill file: bring its imports and its test helpers along
    drill_imports = [seg(dl, n.lineno, n.end_lineno) for n in dtree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                     and not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith(("_check", "solutions")))]
    all_targets = {n2 for it in items for n2 in it["targets"]} | {n2 for v in todo_of.values() for n2 in v}
    helpers = [seg(dl, start_line(n), n.end_lineno) for n in top
               if not (node_names(n) & set(sol_nodes)) and not (node_names(n) & all_targets)]
    ref = ("# imports of the drill file (the checks use them)\n" + "\n".join(drill_imports) + "\n\n" + ref.rstrip() + "\n"
           + ("\n\n# helpers the checks use (defined in the drill file)\n" + "\n\n\n".join(helpers) + "\n" if helpers else ""))
    (REF / f"day{day}.py").write_text(ref, encoding="utf-8")
    ref_list = ", ".join(extra_ref + [f"day{day}"])

    out = [f"@@ day {day}", f"title: {title}", f"part: {PARTS[day]}", "hours: 3–4 小时", "--- goal", "TODO: 一句话写今天学完能做什么。",
           "--- checklist", "- 先清空复习区", "- 读完知识卡片并改动运行", "- 练习区全部通过", ""]
    for n, it in enumerate(items, 1):
        iid = f"d{day}-{n:02d}"
        if it["type"] == "self":
            out += [f"@@ item {iid}", "type: self", f"title: {it['title']}", "level: 1", "--- context", "TODO",
                    "--- prompt", it["prompt"] or "TODO", "--- checklist", "- TODO: 从 prompt 里的 [ ] 清单搬过来", ""]
            continue
        starter = "\n\n".join(seg(dl, start_line(names_defined[n2]), names_defined[n2].end_lineno) for n2 in it["targets"]) if it["targets"] else ""
        solution = "\n\n".join(seg(sl, start_line(sol_nodes[n2]), sol_nodes[n2].end_lineno) for n2 in it["targets"] if n2 in sol_nodes)
        missing = [n2 for n2 in it["targets"] if n2 not in sol_nodes]
        if missing:
            REPORT.append(f"day {day} {iid}: solution has no {missing}")
        typ = "code" if it["type"] in ("code", "demo") else it["type"]
        out += [f"@@ item {iid}", f"type: {typ}", f"title: {it['title']}", "level: 2",
                f"targets: {', '.join(it['targets'])}", f"ref: {ref_list}"]
        if it["merged"]:
            out.append("# merged checks: " + "; ".join(it["merged"]))
        if it["type"] == "demo":
            out.append("# CONVERTER: demo without learner code - turn into a card + a choice item")
        out += ["--- context", "TODO", "--- prompt", it["prompt"] or "TODO", "--- starter", starter or "# TODO",
                "--- solution", solution or "# TODO", "--- tests", "\n\n".join(it["checks"]), "--- hints", "- TODO", ""]
    path = CONTENT / f"day{day}.txt"
    if path.exists() and not FORCE:
        REPORT.append(f"day {day}: {path.name} exists, not overwritten")
    else:
        path.write_text("\n".join(out), encoding="utf-8")
    return len(items)


def todo_owner(todo_of, name):
    return any(name in v for v in todo_of.values())


def main():
    REF.mkdir(parents=True, exist_ok=True)
    total = 0
    for day in range(1, 15):
        n = convert(day)
        total += n
        print(f"day {day}: {n} items")
    print(f"total {total} items")
    for r in REPORT:
        print("  ! " + r)


if __name__ == "__main__":
    main()
