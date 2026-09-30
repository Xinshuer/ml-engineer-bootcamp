"""Runs one exercise: setup code, the learner's code, then the item's checks, in a fresh namespace.

Two ways to use it:
  - in a long-lived worker process (`python app/runner.py --worker`): reads one JSON job per line on stdin,
    writes one JSON result per line. torch is imported once, so a run takes milliseconds instead of seconds.
  - directly: `run_job(job)` (the build validator uses the worker too, so both paths are the same code).

A job: {"setup": str, "code": str, "tests": str, "lang": "zh"|"en", "mode": "check"|"free"}
A result: {"checks": [{"title", "passed", "todo", "message"}], "stdout", "error", "error_line", "seconds"}
"""
import contextlib
import io
import json
import linecache
import os
import re
import sys
import time
import traceback

USER_FILE = "<your code>"
TESTS_FILE = "<tests>"
SETUP_FILE = "<setup>"

MSG = {
    "zh": {"todo": "这里还没写", "setup": "题目的准备代码出错了（这是课程的 bug，请告诉作者）", "no_checks": "这道题没有检查项"},
    "en": {"todo": "not written yet", "setup": "the exercise's setup code failed (a bug in the course; please report it)", "no_checks": "this exercise has no checks"},
}


# ---------------------------------------------------------------- the helpers every exercise can use
class NotWrittenYet(BaseException):
    """What `todo()` raises. Not an Exception (NotImplementedError is a RuntimeError), so a test that wraps the
    learner's call in `except Exception` / `except RuntimeError` cannot swallow an unwritten starter."""


def _helpers(lang):
    import torch

    zh = lang != "en"

    def todo(hint=""):
        raise NotWrittenYet(hint or MSG[lang]["todo"])

    def eq(got, want, tol=1e-4, msg=""):
        """Compare numbers or tensors: the shape first, then the largest absolute difference."""
        g = torch.as_tensor(got)
        w = torch.as_tensor(want)
        if tuple(g.shape) != tuple(w.shape):
            raise AssertionError((f"shape 不对: 你的 {tuple(g.shape)}, 应为 {tuple(w.shape)}. " if zh
                                  else f"wrong shape: yours {tuple(g.shape)}, expected {tuple(w.shape)}. ") + msg)
        d = (g.detach().float().cpu() - w.detach().float().cpu()).abs().max().item() if g.numel() else 0.0
        if not (d <= tol):
            raise AssertionError((f"数值不对: max|差| = {d:.3e} > 容差 {tol}. " if zh
                                  else f"wrong values: max |difference| = {d:.3e} > tolerance {tol}. ") + msg)

    def shape_is(t, s, msg=""):
        got = tuple(t.shape) if hasattr(t, "shape") else tuple(t)
        if got != tuple(s):
            raise AssertionError((f"shape 不对: 你的 {got}, 应为 {tuple(s)}. " if zh
                                  else f"wrong shape: yours {got}, expected {tuple(s)}. ") + msg)

    def true(cond, msg=""):
        if not cond:
            raise AssertionError(msg or ("断言失败" if zh else "assertion failed"))

    def close(a, b, tol=1e-4):
        return abs(float(a) - float(b)) <= tol

    def seed(n=0):
        torch.manual_seed(n)

    def params(m):
        return sum(p.numel() for p in m.parameters())

    def trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    return dict(todo=todo, eq=eq, shape_is=shape_is, true=true, close=close, seed=seed, params=params, trainable=trainable)


_BACKEND_FLAGS = None  # (getter, setter, value at start-up) for backend switches a run may flip
_DYNAMO_DIRTY = False  # the last run may have used torch.compile


def _backend_flags(torch):
    b = torch.backends
    pairs = [(lambda o=o, n=n: getattr(o, n), lambda v, o=o, n=n: setattr(o, n, v))
             for o, n in ((b.cudnn, "benchmark"), (b.cudnn, "deterministic"), (b.cudnn, "allow_tf32"),
                          (b.cuda.matmul, "allow_tf32"))]
    pairs.append((torch.get_float32_matmul_precision, torch.set_float32_matmul_precision))
    out = []
    for get, put in pairs:
        try:
            out.append((get, put, get()))
        except Exception:
            pass
    return out


def _reset_torch(dynamo=False):
    """Undo global state a previous run may have changed. Everything here runs before every run and every
    check, so it only touches what changed (turning deterministic mode off, or resetting dynamo, costs ~40 ms)."""
    global _BACKEND_FLAGS
    import random

    import torch

    if _BACKEND_FLAGS is None:
        _BACKEND_FLAGS = _backend_flags(torch)
    for get, put, start in _BACKEND_FLAGS:
        try:
            if get() != start:  # some of these print deprecation warnings when set
                put(start)
        except Exception:  # e.g. reading the precision after a run mixed the old and new TF32 APIs
            pass
    if dynamo and "torch._dynamo" in sys.modules:  # compiled code and "automatic dynamic" choices outlive a run
        try:
            sys.modules["torch._dynamo"].reset()
        except Exception:
            pass
    random.seed(0)
    if "numpy" in sys.modules:
        sys.modules["numpy"].random.seed(0)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(True)
    try:
        torch.set_default_device(None)  # not "cpu": that installs a Python hook every torch call then goes through
    except Exception:
        pass
    try:
        if torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_WARN_FILTERS = None
# a traceback that a warning prints (e.g. detect_anomaly's "Traceback of forward call") starts in this file:
# those frames are the course's plumbing, not the learner's code, so they are cut from the output
_RUNNER_FRAME = re.compile(r'^ *File "' + re.escape(os.path.abspath(__file__)) + r'", line \d+, in .*\n(?: {4,}.*\n)?',
                           re.M | re.I)


def _reset_warnings():
    """Each run sees warnings like a fresh process would: the start-up filters (a run's setup may add its own
    again) and no memory of warnings an earlier run already showed."""
    global _WARN_FILTERS
    import warnings

    if _WARN_FILTERS is None:
        _WARN_FILTERS = list(warnings.filters)
    warnings.filters[:] = _WARN_FILTERS
    for fn in ("_filters_mutated", "_filters_mutated_lock_held"):  # invalidates the "already shown" registries
        if hasattr(warnings, fn):
            try:
                getattr(warnings, fn)()
                break
            except Exception:
                pass


def _user_line(tb):
    """The last line number inside the learner's code in a traceback, or None."""
    line = None
    for frame, lineno in traceback.walk_tb(tb):
        if frame.f_code.co_filename == USER_FILE:
            line = lineno
    return line


def _format_error(e, tb):
    """`TypeError: ...` plus the learner's lines of the traceback (the rest is noise for a learner)."""
    lines = []
    for fs in traceback.extract_tb(tb):
        if fs.filename == USER_FILE:
            src = (fs.line or "").strip()
            lines.append(f'  line {fs.lineno}, in {fs.name}' + (f"\n    {src}" if src else ""))
    head = f"{type(e).__name__}: {e}"
    return (("\n".join(lines) + "\n") if lines else "") + head


# The "try it" part at the bottom of an exercise (sample data above the function, a call that prints below):
# only the code above this line is graded; the rest runs after the checks, and what it prints or raises is
# reported on its own ("demo"), so a function that is not written yet still shows as "not written yet".
TRY_RE = re.compile(r"^[ \t]*# ---- (?:试一试|Try it)", re.M)


def _run_demo(demo_src, ns, lang):
    out = io.StringIO()
    res = {"stdout": "", "error": None, "line": None}
    _reset_torch()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            exec(compile(demo_src, USER_FILE, "exec"), ns)
        except NotWrittenYet as e:
            res["error"] = str(e) or MSG[lang]["todo"]
            res["line"] = _user_line(e.__traceback__)
        except BaseException as e:  # noqa: BLE001
            res["error"] = _format_error(e, e.__traceback__)
            res["line"] = _user_line(e.__traceback__)
    res["stdout"] = _RUNNER_FRAME.sub("", out.getvalue())[-8000:]
    return res


def run_job(job):
    t0 = time.time()
    lang = job.get("lang", "zh")
    code = job.get("code") or ""
    setup = job.get("setup") or ""
    tests = job.get("tests") or ""
    mode = job.get("mode", "check")
    out = io.StringIO()
    result = {"checks": [], "stdout": "", "error": None, "error_line": None}
    try_at = TRY_RE.search(code) if mode == "check" else None
    demo_src = None
    if try_at:  # keep the editor's line numbers in the try-it part's tracebacks
        demo_src = "\n" * code.count("\n", 0, try_at.start()) + code[try_at.start():]
        code = code[:try_at.start()]

    # make `inspect.getsource` / tracebacks show the learner's lines
    for name, text in ((USER_FILE, code), (TESTS_FILE, tests), (SETUP_FILE, setup)):
        linecache.cache[name] = (len(text), None, [ln + "\n" for ln in text.splitlines()], name)

    global _DYNAMO_DIRTY
    _reset_warnings()
    _reset_torch(dynamo=_DYNAMO_DIRTY)
    _DYNAMO_DIRTY = any(w in s for s in (code, setup, tests) for w in ("compile", "_dynamo"))
    ns = {"__name__": "__main__", "__file__": USER_FILE}
    ns.update(_helpers(lang))
    registered = []

    def check(title):
        def deco(fn):
            registered.append((title, fn))
            return fn
        return deco

    ns["check"] = check
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            exec(compile(setup, SETUP_FILE, "exec"), ns)
        except BaseException as e:  # noqa: BLE001 - report everything, never crash the worker
            result["error"] = MSG[lang]["setup"] + "\n" + "".join(traceback.format_exception(e))[-3000:]
            result["setup_failed"] = True
        main_ok = False
        if not result["error"]:
            try:
                exec(compile(code, USER_FILE, "exec"), ns)
                main_ok = True
            except SyntaxError as e:
                result["error"] = f"SyntaxError: {e.msg}" + (f"\n  line {e.lineno}: {(e.text or '').strip()}" if e.lineno else "")
                result["error_line"] = e.lineno
            except BaseException as e:  # noqa: BLE001
                result["error"] = _format_error(e, e.__traceback__)
                result["error_line"] = _user_line(e.__traceback__)
        if not result["error"] and mode == "check":
            try:
                exec(compile(tests, TESTS_FILE, "exec"), ns)
            except BaseException as e:  # noqa: BLE001
                result["error"] = MSG[lang]["setup"] + "\n" + "".join(traceback.format_exception(e))[-3000:]
                result["setup_failed"] = True
            for title, fn in registered:
                _reset_torch()
                c = {"title": title, "passed": False, "todo": False, "message": ""}
                try:
                    fn()
                    c["passed"] = True
                except (NotWrittenYet, NotImplementedError) as e:
                    c["todo"] = True
                    c["message"] = str(e) or MSG[lang]["todo"]
                except AssertionError as e:
                    c["message"] = str(e)
                    ln = _user_line(e.__traceback__)
                    if ln:
                        c["line"] = ln
                except BaseException as e:  # noqa: BLE001
                    c["message"] = _format_error(e, e.__traceback__)
                    ln = _user_line(e.__traceback__)
                    if ln:
                        c["line"] = ln
                        result["error_line"] = result["error_line"] or ln
                result["checks"].append(c)
            if not registered and not result["error"]:
                result["error"] = MSG[lang]["no_checks"]
    if demo_src is not None and main_ok and not result.get("setup_failed"):
        result["demo"] = _run_demo(demo_src, ns, lang)
    result["stdout"] = _RUNNER_FRAME.sub("", out.getvalue())[-20000:]
    result["seconds"] = round(time.time() - t0, 3)
    return result


def _leaked_mode(torch):
    """True when a run left inference mode on (e.g. `with torch.inference_mode():` inside a generator the learner
    never finished). Collecting garbage closes such generators; if the mode is still on, it cannot be undone here."""
    try:
        if not torch.is_inference_mode_enabled():
            return False
        import gc

        gc.collect()
        return torch.is_inference_mode_enabled()
    except Exception:
        return False


def worker():
    """JSON lines in, JSON lines out. Anything the learner's code (or a C extension) writes to the real
    stdout would corrupt the protocol, so fd 1 is pointed at stderr and the protocol uses a copy of it."""
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    import torch  # noqa: F401 - warm up once
    proto.write(json.dumps({"ready": True, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}) + "\n")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = None
        try:
            job = json.loads(line)
            res = run_job(job)
        except BaseException as e:  # noqa: BLE001
            res = {"checks": [], "stdout": "", "error": f"worker: {type(e).__name__}: {e}", "error_line": None}
        res["id"] = job.get("id") if isinstance(job, dict) else None
        if _leaked_mode(torch):
            res["recycle"] = True  # the pool replaces this process: the next run must not inherit the mode
        proto.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        print(json.dumps(run_job(json.loads(sys.stdin.read())), ensure_ascii=False, indent=1))
