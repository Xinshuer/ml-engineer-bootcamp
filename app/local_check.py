"""Check a workspace file in the terminal with the same checks the page uses:

    python workspace/d4-02.py

(the file calls main() below). Prints one line per check, like the classic drills: [OK] / [XX] / [--].
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.runner import run_job  # noqa: E402

CODE_START = "# ==== your code ===="
CODE_END = "# ==== end of your code ===="


def main(path, lang="zh"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    path = Path(path)
    item_id = path.stem
    runs = ROOT / "app" / "static" / f"runs-{lang}.json"
    if not runs.exists():
        runs = ROOT / "app" / "static" / "runs-zh.json"
    spec = json.loads(runs.read_text(encoding="utf-8")).get(item_id)
    if not spec:
        raise SystemExit(f"{item_id}: no such exercise (was the course rebuilt with other ids?)")
    text = path.read_text(encoding="utf-8")
    code = text.split(CODE_START, 1)[1].split(CODE_END, 1)[0] if CODE_START in text else text
    res = run_job({"setup": spec["setup"], "code": code, "tests": spec["tests"], "lang": lang})
    zh = lang != "en"
    print("=" * 64)
    print(f"  {item_id}")
    print("=" * 64)
    if res.get("stdout"):
        print(res["stdout"].rstrip())
        print("-" * 64)
    if res.get("error"):
        print(("报错:\n" if zh else "Error:\n") + res["error"])
    ok = 0
    for c in res["checks"]:
        mark = "[OK]" if c["passed"] else "[--]" if c["todo"] else "[XX]"
        print(f"{mark} {c['title']}")
        if not c["passed"] and c.get("message"):
            print(f"       -> {c['message']}")
        ok += c["passed"]
    print("-" * 64)
    n = len(res["checks"])
    print(f"  {'通过' if zh else 'passed'} {ok}/{n}    ({res.get('seconds')} s)")
    print("=" * 64)
    raise SystemExit(0 if n and ok == n and not res.get("error") else 1)
