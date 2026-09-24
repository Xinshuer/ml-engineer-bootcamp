"""Day 19 hidden helpers (shared by the Chinese and the English exercises).

saved_bytes(fn, *args, skip=())  measures activation memory on a CPU: the bytes autograd keeps for backward.
FakeGPU(durations, async_=True)  a pretend device with a fake clock, to test a timer without real time.
"""
import torch


def saved_bytes(fn, *args, skip=()):
    """Run fn(*args) and return (its result, the bytes autograd saved for backward while it ran).
    Each storage counts once. Storages shared with the tensors in `skip` (e.g. model.parameters()) are not
    counted: they exist anyway."""
    skip_ptrs = {t.untyped_storage().data_ptr() for t in skip}
    seen = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in skip_ptrs:
            seen[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(*args)
    return out, sum(seen.values())


class FakeGPU:
    """A pretend device: no real time passes, so tests of a timer are exact.

    fn()     does the next piece of work; `durations[i]` is how long call i takes.
             async_=True: the work is only queued and fn() returns at once (like launching a CUDA kernel).
             async_=False: the work is done before fn() returns (like code on the CPU).
    sync()   waits until the queue is empty: the fake clock jumps forward by the queued work.
    clock()  reads the fake time (use it where you would use time.perf_counter).
    """

    def __init__(self, durations, async_=True):
        self.durations = list(durations)
        self.async_ = async_
        self.now = 0.0
        self.pending = 0.0
        self.calls = 0
        self.syncs = 0

    def fn(self):
        d = self.durations[self.calls] if self.calls < len(self.durations) else 1000.0
        self.calls += 1
        if self.async_:
            self.pending += d
        else:
            self.now += d

    def sync(self):
        self.syncs += 1
        self.now += self.pending
        self.pending = 0.0

    def clock(self):
        return self.now
