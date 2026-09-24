"""The course runs in your browser, the code runs here with real PyTorch.

    python serve.py                 # http://127.0.0.1:8765 opens in your browser
    python serve.py --port 9000 --no-browser

Only this computer can reach the server (it listens on 127.0.0.1). Every request from the page carries a
token that is generated at start-up and only written into the page, so another website cannot make this
server run code on your computer.
"""
import argparse
import json
import re
import secrets
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from app.pool import Worker  # noqa: E402

STATIC = ROOT / "app" / "static"
WORKSPACE = ROOT / "workspace"
PROGRESS = ROOT / "progress.json"
TOKEN = secrets.token_urlsafe(24)
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}
ID_RE = re.compile(r"^d\d+-(\d\d|c\d+|[a-z][\w-]*)$")
CODE_START = "# ==== your code ===="
CODE_END = "# ==== end of your code ===="


class Course:
    def __init__(self):
        self.runs = {}
        self.titles = {}
        for lang in ("zh", "en"):
            p = STATIC / f"runs-{lang}.json"
            if p.exists():
                self.runs[lang] = json.loads(p.read_text(encoding="utf-8"))
        if "zh" not in self.runs:
            raise SystemExit("The course is not built yet. Run:  python tools/build.py")

    def run_spec(self, lang, item_id):
        return (self.runs.get(lang) or self.runs["zh"]).get(item_id)


COURSE = None
WORKER = Worker()
WORKER_READY = threading.Event()


def start_worker():
    try:
        WORKER.start()
    finally:
        WORKER_READY.set()


def workspace_file(item_id, code, lang):
    head = ("# {id}  ML Engineer Bootcamp\n# Check it (from the course folder):  python workspace/{id}.py\n"
            if lang == "en" else "# {id}  ML 工程师训练营\n# 判题（在课程文件夹里运行）:  python workspace/{id}.py\n").format(id=item_id)
    return (f"{head}\n{CODE_START}\n{code.rstrip()}\n{CODE_END}\n\n"
            f'if __name__ == "__main__":\n'
            f"    import sys, pathlib\n"
            f"    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))\n"
            f"    from app.local_check import main\n"
            f'    main(__file__, "{lang}")\n')


def code_of(text):
    if CODE_START in text and CODE_END in text:
        return text.split(CODE_START, 1)[1].split(CODE_END, 1)[0].strip("\n") + "\n"
    return text


class Handler(BaseHTTPRequestHandler):
    server_version = "mleb"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _host_ok(self):
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        return host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _send(self, status, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        return self._host_ok() and secrets.compare_digest(self.headers.get("X-CC-Token", ""), TOKEN)

    def do_GET(self):
        if not self._host_ok():
            return self._send(HTTPStatus.FORBIDDEN, {"error": "bad host"})
        path = self.path.split("?")[0]
        if path == "/favicon.ico":
            return self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        if path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_text(encoding="utf-8").replace("{{TOKEN}}", TOKEN)
            return self._send(HTTPStatus.OK, html.encode("utf-8"), MIME[".html"])
        if path.startswith("/static/"):
            f = (STATIC / path[len("/static/"):]).resolve()
            if STATIC.resolve() not in f.parents or not f.is_file() or f.name.startswith("runs-"):
                return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return self._send(HTTPStatus.OK, f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
        if path.startswith("/api/"):
            if not self._authorized():
                return self._send(HTTPStatus.FORBIDDEN, {"error": "missing token"})
            if path == "/api/status":
                WORKER_READY.wait(timeout=1)
                info = WORKER.info or {}
                return self._send(HTTPStatus.OK, {"ready": bool(info.get("ready")), **info, "python": sys.version.split()[0]})
            if path == "/api/progress":
                data = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
                return self._send(HTTPStatus.OK, data)
        return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            return self._send(HTTPStatus.FORBIDDEN, {"error": "missing token"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > 5_000_000:
            return self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "too large"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(HTTPStatus.BAD_REQUEST, {"error": "bad json"})
        path = self.path.split("?")[0]
        if path == "/api/progress":
            tmp = PROGRESS.with_suffix(".tmp")
            tmp.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(PROGRESS)
            return self._send(HTTPStatus.OK, {"ok": True})
        item_id = str(body.get("id", ""))
        lang = "en" if body.get("lang") == "en" else "zh"
        if not ID_RE.match(item_id):
            return self._send(HTTPStatus.BAD_REQUEST, {"error": "bad id"})
        spec = COURSE.run_spec(lang, item_id)
        if path in ("/api/run", "/api/free"):
            if not spec:
                return self._send(HTTPStatus.NOT_FOUND, {"error": f"no runnable item {item_id}"})
            WORKER_READY.wait(timeout=240)
            job = {"setup": spec["setup"], "code": str(body.get("code", "")), "lang": lang,
                   "tests": spec["tests"] if path == "/api/run" else "", "mode": "check" if path == "/api/run" else "free"}
            return self._send(HTTPStatus.OK, WORKER.run(job, timeout=spec.get("timeout", 60)))
        if path == "/api/save":
            WORKSPACE.mkdir(exist_ok=True)
            f = WORKSPACE / f"{item_id}.py"
            f.write_text(workspace_file(item_id, str(body.get("code", "")), lang), encoding="utf-8")
            return self._send(HTTPStatus.OK, {"path": f"workspace/{item_id}.py", "command": f"python workspace/{item_id}.py"})
        if path == "/api/load":
            f = WORKSPACE / f"{item_id}.py"
            if not f.exists():
                return self._send(HTTPStatus.NOT_FOUND, {"error": "no file"})
            return self._send(HTTPStatus.OK, {"code": code_of(f.read_text(encoding="utf-8"))})
        return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})


def main():
    global COURSE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    COURSE = Course()
    threading.Thread(target=start_worker, daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"ML Engineer Bootcamp / ML 工程师训练营  ->  {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        WORKER.stop()


if __name__ == "__main__":
    main()
