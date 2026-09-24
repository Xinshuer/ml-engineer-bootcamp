"""Day 24 参考实现：实验可复现、模型可追溯。
Day 24 reference code: reproducible runs and traceable models.

练习的 tests 和卡片的例子会用到这里的函数和类；每道题要写的名字（targets）会在运行前被删掉。
The exercises' tests and the cards use the functions and classes below; each item's targets are removed before it runs.
"""
import copy
import hashlib
import json
import os
import platform
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- data fingerprints / 数据指纹
def file_sha256(path, chunk_size=1 << 20):
    """一个文件的字节的 sha256，分块读（10 GB 的文件也不用一次装进内存）。
    sha256 of one file's bytes, read in chunks (a 10 GB file never has to fit in memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def data_fingerprint(paths):
    """一组文件的指纹：每个文件的哈希排好序、用换行连起来，再算一次 sha256。只看内容，不看顺序和文件名。
    One hash for a set of files: sort the per-file hashes, join them with newlines, hash again.
    Only the bytes count: not the order of `paths`, not the file names or folders."""
    hashes = sorted(file_sha256(p) for p in paths)
    return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()


def write_files(folder, files):
    """files: {文件名: bytes 或 str} -> 完整路径的 list（按 dict 的顺序）。给卡片和检查用。
    files: {name: bytes or str} -> list of full paths (in the dict's order). For cards and checks."""
    paths = []
    for name, content in files.items():
        path = os.path.join(folder, name)
        with open(path, "wb") as f:
            f.write(content.encode("utf-8") if isinstance(content, str) else content)
        paths.append(path)
    return paths


# ---------------------------------------------------------------- the run record / 实验记录
def library_versions():
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}


def make_run_record(run_id, config, git_commit, data_paths, seed, metrics):
    return {
        "run_id": run_id,
        "config": copy.deepcopy(config),          # 拷贝：之后改 config 不影响记录 / a copy, not a reference
        "git_commit": git_commit,
        "data_fingerprint": data_fingerprint(data_paths),
        "seed": seed,
        "versions": library_versions(),
        "metrics": {k: float(v) for k, v in metrics.items()},   # 0 维张量、numpy float32 -> float
    }


def save_run(record, folder):
    path = os.path.join(folder, record["run_id"] + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return path


def load_runs(folder):
    runs = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                runs.append(json.load(f))
    return runs


# ---------------------------------------------------------------- comparing runs / 比较实验
_FP = "5d1e0c3b7a9f24e6c8b1d3f5a7e9c2b4d6f8a1c3e5b7d9f2a4c6e8b1d3f5a7e9"
_COMMIT = "3f9a2c1d0b7e4f6a8c5d2e1f0a9b8c7d6e5f4a3b"


def sample_runs():
    """四份已经跑完的实验记录（每次调用都是新的副本）。run-03 的 loss 炸成了 nan，还没评估就停了。
    Four finished run records (fresh copies on every call). run-03 blew up (nan loss) before evaluation."""
    base = {"model": "tiny-cnn", "lr": 0.001, "batch_size": 32, "epochs": 10}
    plan = [
        ("run-01", {}, {"val_acc": 0.842, "val_loss": 0.47}),
        ("run-02", {"lr": 0.003}, {"val_acc": 0.861, "val_loss": 0.44}),
        ("run-03", {"lr": 0.03}, {}),
        ("run-04", {"lr": 0.003, "batch_size": 64, "warmup": 100}, {"val_acc": 0.875, "val_loss": 0.45}),
    ]
    runs = []
    for run_id, changes, metrics in plan:
        runs.append({
            "run_id": run_id,
            "config": {**base, **changes},
            "git_commit": _COMMIT,
            "data_fingerprint": _FP,
            "seed": 0,
            "versions": {"python": "3.12.7", "torch": "2.7.1", "numpy": "2.2.6"},
            "metrics": dict(metrics),
        })
    return runs


def best_run(runs, metric, higher_is_better=True):
    have = [r for r in runs if metric in r["metrics"]]
    if not have:
        raise ValueError(f"no run has the metric {metric!r}")
    pick = max if higher_is_better else min
    return pick(have, key=lambda r: r["metrics"][metric])


def config_diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


# ---------------------------------------------------------------- a tiny model registry / 一个小小的模型注册表
class JsonStore:
    """注册表就是 folder 里的一个 registry.json。每个方法都是「读文件 -> 改 -> 写回」，不在内存里留状态，
    所以新建一个对象（或者另一个进程）看到的是同一个状态。
    The registry is one registry.json inside `folder`. Every method loads, changes, saves: nothing is kept
    only in memory, so a new object (another process) sees the same state."""

    def __init__(self, folder, metric="val_acc", higher_is_better=True):
        self.folder = folder
        self.path = os.path.join(folder, "registry.json")
        self.metric = metric
        self.higher_is_better = higher_is_better

    def _load(self):
        if not os.path.exists(self.path):
            return {"versions": [], "production": None, "history": []}
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)          # 原子写：写到一半崩了也不会留下坏文件 / atomic (day 20)


class BasicRegistry(JsonStore):
    """register / get / production / promote（promote 没有门槛）。 promote here has no gate."""

    def register(self, run_id, metrics):
        data = self._load()
        version = len(data["versions"]) + 1
        data["versions"].append({"version": version, "run_id": run_id, "metrics": dict(metrics)})
        self._save(data)
        return version

    def get(self, version):
        for entry in self._load()["versions"]:
            if entry["version"] == version:
                return entry
        raise ValueError(f"no version {version} in the registry")

    def production(self):
        return self._load()["production"]

    def promote(self, version):
        self.get(version)                    # 不存在 -> ValueError / unknown version -> ValueError
        data = self._load()
        data["production"] = version
        data["history"].append(version)
        self._save(data)


class Registry(BasicRegistry):
    """promote 带门槛（不比线上差才能上），加上 rollback。 promote with a gate, plus rollback."""

    def promote(self, version):
        new = self.get(version)["metrics"][self.metric]
        data = self._load()
        current = data["production"]
        if current is not None:
            old = self.get(current)["metrics"][self.metric]
            worse = new < old if self.higher_is_better else new > old
            if worse:
                raise ValueError(f"v{version} {self.metric}={new} < production v{current} {self.metric}={old}"
                                 if self.higher_is_better else
                                 f"v{version} {self.metric}={new} > production v{current} {self.metric}={old}")
        data["production"] = version
        data["history"].append(version)
        self._save(data)

    def rollback(self):
        data = self._load()
        if len(data["history"]) < 2:
            raise ValueError("nothing to roll back to")
        data["history"].pop()
        data["production"] = data["history"][-1]
        self._save(data)
        return data["production"]
