"""Parse the course (content/dayN.txt in Chinese, content/en/dayN.txt in English), check that the two
languages match item for item, validate every exercise with real PyTorch, and write app/static/course.js
and app/static/course-en.js for the page.

    python tools/build.py              # everything
    python tools/build.py --day 4,5    # validate only these days (still parses all; writes nothing)
    python tools/build.py --no-run     # parse + lint only
    python tools/build.py --show d4-02 # print the runs of one item

Validation of each runnable item, in both languages:
  - the reference solution passes every check
  - the starter code fails at least one check (a starter that already passes teaches nothing)
  - an empty answer fails too (proves the item's `targets:` are removed from the hidden setup)
"""
import ast
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.pool import Pool  # noqa: E402

CONTENT = ROOT / "content"
REF = CONTENT / "ref"
STATIC = ROOT / "app" / "static"
PRELUDE = "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
RUNNABLE = {"code", "fix", "fill", "order"}
TYPES = RUNNABLE | {"choice", "predict", "self", "local"}
PROBLEMS = []


def bad(where, msg):
    PROBLEMS.append(f"{where}: {msg}")


# ---------------------------------------------------------------- parse
def parse_file(path, days):
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    day = None
    for block in re.split(r"^@@ ", text, flags=re.M)[1:]:
        lines = block.split("\n")
        head = lines[0].split()
        kind, bid = head[0], (head[1] if len(head) > 1 else None)
        obj = {"kind": kind, "id": bid}
        section, buf = None, []

        def flush():
            if section:
                obj[section] = re.sub(r"^\n+|\s+$", "", "\n".join(buf))

        for line in lines[1:]:
            m = re.match(r"^--- (\w+)\s*$", line)
            if m:
                flush()
                section, buf = m.group(1), []
                continue
            late = re.match(r"^(answer|verify|placeholder):\s*(.*)$", line)
            if late and section in (None, "options", "explain", "hints"):
                obj[late.group(1)] = late.group(2).strip()
                continue
            if section:
                buf.append(line)
                continue
            kv = re.match(r"^(\w+):\s*(.*)$", line)
            if kv:
                obj[kv.group(1)] = kv.group(2).strip()
        flush()
        if kind == "day":
            day = {**obj, "id": int(bid), "checklist": as_list(obj.get("checklist")), "concepts": [], "items": [], "file": str(path)}
            day.pop("kind", None)
            days.append(day)
        elif kind == "concept":
            n = len([c for c in day["concepts"] if not c.get("named")]) + 1
            obj["id"] = f"d{day['id']}-{bid}" if bid else f"d{day['id']}-c{n}"
            obj["named"] = bool(bid)
            day["concepts"].append(normalize(obj, day))
        elif kind == "item":
            if day is None:
                bad(f"{path.name}", "an item before the @@ day block")
                continue
            day["items"].append(normalize(obj, day))
        else:
            bad(f"{path.name}", f"unknown block @@ {kind}")


def as_list(s):
    return [x.strip() for x in re.split(r"^- ", s, flags=re.M) if x.strip()] if s else []


def normalize(o, day):
    out = dict(o)
    out.pop("kind", None)
    for k in ("options", "hints", "checklist"):
        if k in o:
            out[k] = as_list(o[k])
    if "answer" in o:
        out["answer"] = [int(x) - 1 for x in o["answer"].split(",")]
    if "level" in o:
        out["level"] = int(o["level"])
    if "timeout" in o:
        out["timeout"] = int(o["timeout"])
    out["exam"] = o.get("exam") == "yes"
    out["error_demo"] = o.get("error_demo") == "yes"
    out["targets"] = [t.strip() for t in o.get("targets", "").split(",") if t.strip()]
    out["ref"] = [t.strip() for t in o.get("ref", day.get("ref", "")).split(",") if t.strip()]
    if o.get("type") == "fill" and "template" in o:
        out["solution"] = re.sub(r"\[\[(.*?)\]\]", r"\1", o["template"])
        out["starter"] = re.sub(r"\[\[(.*?)\]\]", "", o["template"])
        out["fills"] = re.findall(r"\[\[(.*?)\]\]", o["template"])
    if o.get("type") == "order" and "lines" in o:
        out["lines"] = o["lines"].split("\n")
        out["distractors"] = o["distractors"].split("\n") if o.get("distractors") else []
        out["solution"] = o["lines"]
    return out


# ---------------------------------------------------------------- hidden setup = prelude + reference modules - targets
_REF_CACHE = {}


def ref_source(name):
    if name not in _REF_CACHE:
        p = REF / f"{name}.py"
        if not p.exists():
            raise KeyError(name)
        _REF_CACHE[name] = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    return _REF_CACHE[name]


def resolve_setup(block, where):
    """prelude + the reference modules + the item's own --- setup, then the names the learner writes are
    removed again, so an empty answer fails (the reference versions exist only while the setup runs, e.g.
    as the base class of another reference class)."""
    parts = [PRELUDE]
    for name in block.get("ref", []):
        try:
            parts.append(f"# ---- reference: {name}\n" + ref_source(name))
        except KeyError:
            bad(where, f"ref: no content/ref/{name}.py")
    if block.get("setup"):
        parts.append(block["setup"])
    if block.get("targets"):
        parts.append(f"for _name in {block['targets']!r}:\n    globals().pop(_name, None)\ndel _name")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- cards, context, checks, parity
def checks_of(it):
    return re.findall(r'^@check\(\s*[rf]?(["\'])(.*?)\1\s*\)', it.get("tests", ""), flags=re.M)


def prepare(days, lang):
    by_day = {d["id"]: d for d in days}
    ids = set()
    for d in days:
        for blk in d["concepts"] + d["items"]:
            if blk["id"] in ids:
                bad(blk["id"], "duplicate id")
            ids.add(blk["id"])
        for it in d["items"]:
            where = f"{lang} {it['id']}"
            if not re.match(rf"^d{d['id']}-\d\d$", it["id"] or ""):
                bad(where, f"item ids on day {d['id']} look like d{d['id']}-01")
            if it.get("type") not in TYPES:
                bad(where, f"unknown type {it.get('type')}")
            if not it.get("title"):
                bad(where, "no title")
            ctx = it.get("context", "")
            if not ctx.strip():
                bad(where, "empty or missing --- context")
            elif len(ctx) > (600 if lang == "en" else 280):
                bad(where, f"--- context is {len(ctx)} characters (max {600 if lang == 'en' else 280})")
            it["cardIds"] = []
            for ref in [s.strip() for s in it.pop("cards", "").split(";") if s.strip()]:
                m = re.match(r"^d(\d+):\s*(.+)$", ref)
                dd, title = (int(m.group(1)), m.group(2).strip()) if m else (d["id"], ref)
                c = next((c for c in by_day.get(dd, {}).get("concepts", []) if c["title"] == title), None)
                if not c:
                    bad(where, f'cards: no card titled "{title}" on day {dd}')
                elif dd > d["id"]:
                    bad(where, f'cards: "{ref}" is on a later day')
                elif c["id"] not in it["cardIds"]:
                    it["cardIds"].append(c["id"])
            if it.get("type") in RUNNABLE:
                it["checks"] = [{"label": t[1]} for t in checks_of(it)]
                if not it["checks"]:
                    bad(where, "no @check(...) in --- tests")
                for k in ("starter", "solution", "tests"):
                    if not it.get(k):
                        bad(where, f"missing --- {k}")
            if it.get("type") == "choice":
                n = len(it.get("options", []))
                if not n or not it.get("answer") or any(a < 0 or a >= n for a in it["answer"]):
                    bad(where, "choice needs --- options and a valid answer:")
            if it.get("type") == "self" and not it.get("checklist"):
                bad(where, "self needs --- checklist")
            if it.get("type") == "local":
                try:
                    re.compile(it.get("verify", ""))
                except re.error as e:
                    bad(where, f"verify is not a valid regex: {e}")
                if not it.get("verify"):
                    bad(where, "local needs verify:")


def parity(zh, en):
    zd = {d["id"]: d for d in zh}
    for d in en:
        z = zd.get(d["id"])
        if not z:
            bad(f"en day {d['id']}", "no Chinese day with this number")
            continue
        if len(d["checklist"]) != len(z["checklist"]):
            bad(f"en day {d['id']}", "checklist length differs from Chinese")
        if len(d["concepts"]) != len(z["concepts"]):
            bad(f"en day {d['id']}", f"{len(d['concepts'])} cards, Chinese has {len(z['concepts'])}")
        for c, zc in zip(d["concepts"], z["concepts"]):
            if c["id"] != zc["id"]:
                bad(c["id"], f"card id differs from Chinese {zc['id']}")
            if bool(c.get("code")) != bool(zc.get("code")):
                bad(c["id"], "one language has example code, the other not")
            if c.get("ref") != zc.get("ref"):
                bad(c["id"], "ref differs from Chinese")
        if len(d["items"]) != len(z["items"]):
            bad(f"en day {d['id']}", f"{len(d['items'])} items, Chinese has {len(z['items'])}")
        for it, zi in zip(d["items"], z["items"]):
            if it["id"] != zi["id"]:
                bad(it["id"], f"item id differs from Chinese {zi['id']}")
                continue
            for k in ("type", "level", "exam", "answer", "cardIds", "targets", "ref", "timeout"):
                if it.get(k) != zi.get(k):
                    bad(f"en {it['id']}", f"{k} is {it.get(k)!r}, Chinese has {zi.get(k)!r}")
            for k in ("options", "fills", "lines", "distractors", "checks", "hints", "checklist"):
                if len(it.get(k) or []) != len(zi.get(k) or []):
                    bad(f"en {it['id']}", f"{k}: {len(it.get(k) or [])}, Chinese has {len(zi.get(k) or [])}")
            for k in ("starter", "solution", "tests", "code", "template", "context", "explain", "verify", "setup"):
                if bool(it.get(k)) != bool(zi.get(k)):
                    bad(f"en {it['id']}", f"--- {k} present in one language only")


# ---------------------------------------------------------------- validate with real PyTorch
def all_passed(r):
    return not r.get("error") and not r.get("timeout") and r.get("checks") and all(c["passed"] for c in r["checks"])


def validate(days, lang, pool, only, show):
    jobs = []
    for d in days:
        if only and d["id"] not in only:
            continue
        for c in d["concepts"]:
            if c.get("code"):
                c["_setup"] = resolve_setup(c, f"{lang} {c['id']}")
                jobs.append(("concept", c, None))
        for it in d["items"]:
            if it.get("type") in RUNNABLE or it.get("type") == "predict":
                it["_setup"] = resolve_setup(it, f"{lang} {it['id']}")
                if it["type"] == "predict":
                    jobs.append(("predict", it, None))
                else:
                    jobs.append(("solution", it, it.get("solution")))
                    jobs.append(("starter", it, it.get("starter")))
                    if it["targets"]:
                        jobs.append(("blank", it, ""))

    def one(job):
        kind, blk, code = job
        t = blk.get("timeout", 60)
        if kind in ("concept", "predict"):
            r = pool.run({"setup": blk["_setup"], "code": blk["code"], "mode": "free", "lang": lang}, t)
        else:
            r = pool.run({"setup": blk["_setup"], "code": code, "tests": blk["tests"], "lang": lang}, t)
        return kind, blk, r

    t0 = time.time()
    with ThreadPoolExecutor(len(pool.workers)) as ex:
        for kind, blk, r in ex.map(one, jobs):
            where = f"{lang} {blk['id']}"
            if show and blk["id"] in show:
                print(f"--- {where} [{kind}] {r.get('seconds')}s\n{json.dumps(r, ensure_ascii=False, indent=1)[:4000]}")
            if r.get("timeout"):
                bad(where, f"{kind}: timed out after {blk.get('timeout', 60)} s")
                continue
            if r.get("setup_failed"):
                bad(where, f"{kind}: setup/tests failed: {(r.get('error') or '')[-600:]}")
                continue
            if kind == "concept":
                if r.get("error") and not blk.get("error_demo"):
                    bad(where, f"example code fails: {r['error'][-300:]}")
            elif kind == "predict":
                if r.get("error"):
                    bad(where, f"predict code fails: {r['error'][-300:]}")
                blk["expected"] = r.get("stdout", "").rstrip("\n")
            elif kind == "solution":
                if not all_passed(r):
                    fails = [f"{c['title']}: {c['message'][:160]}" for c in r.get("checks", []) if not c["passed"]]
                    bad(where, f"solution does not pass: {(r.get('error') or '')[-300:]} {fails}")
                blk["_sol_seconds"] = r.get("seconds")
            elif kind in ("starter", "blank"):
                if all_passed(r):
                    bad(where, f"the {kind} already passes every check")
    return len(jobs), time.time() - t0


# ---------------------------------------------------------------- emit
def public(days):
    """What the page gets: everything except build-only fields."""
    out = []
    for d in days:
        dd = {k: v for k, v in d.items() if k not in ("file",)}
        dd["concepts"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in d["concepts"]]
        dd["items"] = [{k: v for k, v in it.items() if not k.startswith("_")} for it in d["items"]]
        out.append(dd)
    return out


def main():
    args = sys.argv[1:]
    only = None
    show = set()
    if "--day" in args:
        only = {int(x) for x in args[args.index("--day") + 1].split(",")}
    if "--show" in args:
        show = set(args[args.index("--show") + 1].split(","))
    run = "--no-run" not in args

    zh, en = [], []
    day_files = lambda d: sorted(d.glob("day*.txt"), key=lambda p: int(re.search(r"\d+", p.name).group()))  # noqa: E731
    for f in day_files(CONTENT):
        parse_file(f, zh)
    for f in day_files(CONTENT / "en"):
        parse_file(f, en)
    prepare(zh, "zh")
    prepare(en, "en")
    en_complete = bool(en) and len(en) == len(zh)
    if en:
        parity(zh, en)
    n_items = sum(len(d["items"]) for d in zh)
    print(f"parsed: {len(zh)} days, {n_items} items, {sum(len(d['concepts']) for d in zh)} cards (zh); "
          f"{len(en)} days (en)")

    if run:
        workers = int(os.environ.get("BUILD_WORKERS", "6"))
        pool = Pool(workers)
        try:
            n, secs = validate(zh, "zh", pool, only, show)
            print(f"zh: {n} runs in {secs:.0f} s")
            if en:
                n, secs = validate(en, "en", pool, only, show)
                print(f"en: {n} runs in {secs:.0f} s")
        finally:
            pool.close()
    else:
        for days in (zh, en):
            for d in days:
                for blk in d["concepts"] + d["items"]:
                    if blk.get("code") or blk.get("type") in RUNNABLE:
                        blk["_setup"] = resolve_setup(blk, blk["id"])

    if only:
        # several people (or agents) edit different days at once: report only the days asked for
        def day_of(p):
            m = re.search(r"\bd(\d+)-|\bday ?(\d+)\b", p)
            return int(m.group(1) or m.group(2)) if m else None
        PROBLEMS[:] = [p for p in PROBLEMS if day_of(p) is None or day_of(p) in only]
    if PROBLEMS:
        print(f"\n{len(PROBLEMS)} problem(s):")
        for p in PROBLEMS:
            print("  " + p)
        raise SystemExit(1)
    if only or not run:
        print("ok (nothing written: partial run)" if only else "ok (nothing written: --no-run)")
        return
    STATIC.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for lang, days in (("zh", zh), ("en", en if en_complete else None)):
        if days is None:
            continue
        # the server needs the hidden setup + tests to run an item; the page needs everything else
        server = {it["id"]: {"setup": it.get("_setup", ""), "tests": it.get("tests", ""), "timeout": it.get("timeout", 60)}
                  for d in days for it in d["items"] if it.get("type") in RUNNABLE or it.get("type") == "predict"}
        server.update({c["id"]: {"setup": c.get("_setup", ""), "tests": "", "timeout": c.get("timeout", 60)}
                       for d in days for c in d["concepts"] if c.get("code")})
        (STATIC / f"runs-{lang}.json").write_text(json.dumps(server, ensure_ascii=False), encoding="utf-8")
        var = "CC_COURSE" if lang == "zh" else "CC_COURSE_EN"
        body = json.dumps({"builtAt": stamp, "days": public(days)}, ensure_ascii=False)
        (STATIC / ("course.js" if lang == "zh" else "course-en.js")).write_text(f"window.{var} = {body};\n", encoding="utf-8")
    print(f"all exercises valid; wrote app/static/course.js{' + course-en.js' if en_complete else ''}")


if __name__ == "__main__":
    main()
