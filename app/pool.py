"""Long-lived runner processes (see runner.py). A run that goes over its time limit kills the process and a
fresh one takes its place, so an endless loop in the learner's code never hangs the app."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "runner.py"


class Worker:
    def __init__(self, python=None):
        self.python = python or sys.executable
        self.proc = None
        self.lines = None
        self.info = {}
        self.lock = threading.Lock()
        self.jobs = 0

    def start(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen([self.python, "-u", str(RUNNER), "--worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
                                     encoding="utf-8", bufsize=1, creationflags=flags)
        self.lines = queue.Queue()
        proc = self.proc

        def pump():
            for line in proc.stdout:
                self.lines.put(line)
            self.lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        first = self.lines.get(timeout=180)
        if not first:
            raise RuntimeError("the runner process exited while starting (is torch installed for this Python?)")
        self.info = json.loads(first)
        self.jobs = 0

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None

    def run(self, job, timeout=30):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None or self.jobs >= 300:
                self.stop()
                self.start()
            t0 = time.time()
            self.proc.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            try:
                line = self.lines.get(timeout=timeout)
            except queue.Empty:
                self.stop()
                return {"timeout": True, "checks": [], "stdout": "", "error_line": None,
                        "seconds": round(time.time() - t0, 1)}
            if line is None:
                self.stop()
                return {"crashed": True, "checks": [], "stdout": "", "error_line": None}
            self.jobs += 1
            return json.loads(line)


class Pool:
    """N workers for the build validator (runs are independent, so they can go in parallel)."""

    def __init__(self, n, python=None):
        self.workers = [Worker(python) for _ in range(n)]
        self.free = queue.Queue()
        for w in self.workers:
            self.free.put(w)

    def run(self, job, timeout=30):
        w = self.free.get()
        try:
            return w.run(job, timeout)
        finally:
            self.free.put(w)

    def close(self):
        for w in self.workers:
            w.stop()
