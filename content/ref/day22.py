"""Day 22 参考实现：把模型做成服务。
Day 22 reference code: serving a model.

练习的 tests 和卡片的例子会用到这里的东西；每道题要写的名字（targets）会在运行前被删掉。
The exercises' tests and the cards use what is defined here; each item's target names are removed before it runs.

    模型和加载器 / model and loaders   FEATURES, LABELS, MAX_ROWS, TinyClassifier, make_model, SpyModel, CountingLoader
    请求格式 / request format          Row, PredictRequest
    服务 / service                     create_app, reference_probs
    预热 / warm-up                     FakeClock, SlowStartModel, warm_up
    状态码 / status codes              ServiceState, create_status_app
    排队 / queue                       Ticket, RequestQueue
    延迟 / latency                     latency_report
    微批 / micro-batching              BatchRecorder, MicroBatcher
"""
import math
import warnings
from typing import Annotated

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat, ValidationError  # noqa: F401 - used by the exercises

with warnings.catch_warnings():
    # starlette 提醒以后换用 httpx2；和今天的内容无关，不让它出现在输出里
    # starlette asks to move to httpx2 one day; unrelated to today, so keep it out of the output
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient  # noqa: F401 - used by the exercises

FEATURES = 4                                   # 每行 4 个数 / 4 numbers per row
LABELS = ["setosa", "versicolor", "virginica"]  # 3 个类别 / 3 classes
MAX_ROWS = 32                                  # 一个请求最多 32 行 / at most 32 rows per request


# ---------------------------------------------------------------- model and loaders / 模型和加载器
class TinyClassifier(nn.Module):
    """4 个数 -> 3 个类别的 logits。带 Dropout，所以 train() 和 eval() 的输出不一样。
    4 numbers -> 3 class logits. Has Dropout, so train() and eval() give different outputs."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURES, 16)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, len(LABELS))

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def _fixed_weights(m):
    g = torch.Generator().manual_seed(22)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return m


def make_model():
    """每次都返回同一组权重。处在 train 模式，和刚建好（或刚 load_state_dict 完）的 nn.Module 一样。
    Always the same weights. In train mode, like any freshly built (or freshly loaded) nn.Module."""
    return _fixed_weights(TinyClassifier())


class SpyModel(TinyClassifier):
    """和 make_model() 一样的模型，另外记下每次 forward 时的情况（log 里一项一个 dict）：
    shape、training（是否 train 模式）、inference（是否在 inference_mode 里）、grad（梯度开没开）。
    The same model as make_model(), and it records every forward call in `log`: shape, training,
    inference (inside inference_mode?), grad (is autograd on?)."""

    def __init__(self):
        super().__init__()
        _fixed_weights(self)
        self.log = []

    def forward(self, x):
        self.log.append({"shape": tuple(x.shape), "training": self.training,
                         "inference": torch.is_inference_mode_enabled(), "grad": torch.is_grad_enabled()})
        return super().forward(x)


class CountingLoader:
    """load_model 的替身：每调用一次就「从硬盘加载」一个新的 SpyModel，并记下次数（calls）和模型（models）。
    A stand-in for load_model: every call "loads" a new SpyModel and records how often (calls) and what (models)."""

    def __init__(self):
        self.calls = 0
        self.models = []

    def __call__(self):
        self.calls += 1
        m = SpyModel()
        self.models.append(m)
        return m


def reference_probs(rows):
    """参考答案：eval 模式下的 softmax 概率，(n, 3)。 The reference: softmax probabilities in eval mode, (n, 3)."""
    m = make_model().eval()
    with torch.inference_mode():
        return m(torch.tensor(rows, dtype=torch.float32)).softmax(-1)


# ---------------------------------------------------------------- request format / 请求格式
Row = Annotated[list[FiniteFloat], Field(min_length=FEATURES, max_length=FEATURES)]


class PredictRequest(BaseModel):
    rows: list[Row] = Field(min_length=1, max_length=MAX_ROWS)


# ---------------------------------------------------------------- the service / 服务
def create_app(load_model):
    model = load_model()                        # 启动时加载一次 / load once, at start-up
    model.eval()
    with torch.inference_mode():                # 预热 / warm-up
        model(torch.zeros(1, FEATURES))
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(req: PredictRequest):
        x = torch.tensor(req.rows, dtype=torch.float32)
        with torch.inference_mode():
            probs = model(x).softmax(-1)
        return {"labels": [LABELS[i] for i in probs.argmax(-1).tolist()], "probs": probs.tolist()}

    return app


# ---------------------------------------------------------------- warm-up / 预热
class FakeClock:
    """假时钟，单位毫秒。它不会自己走：只有 advance(ms) 或直接改 .now 才会变。测试里用它代替真实时间，不用 sleep。
    A fake clock in milliseconds. It never moves by itself: only advance(ms) or setting .now changes it."""

    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, ms):
        self.now += ms


class SlowStartModel(nn.Module):
    """模拟 GPU 上「第一次特别慢」：每遇到一种新的 batch 大小，那一次 forward 多花 compile_ms 毫秒
    （像 torch.compile 为新形状编译、cuDNN 为新形状挑算法）。平时一次 forward 花 run_ms(B) = 2 + 0.25·B 毫秒。
    时间加在假时钟上，不真的等。log 和 SpyModel 一样。
    Simulates slow first calls on a GPU: the first forward for each new batch size costs compile_ms extra
    (like torch.compile or cuDNN autotuning for a new shape). A normal forward costs run_ms(B) = 2 + 0.25*B ms.
    Time is added to the fake clock; nothing really waits. `log` is like SpyModel's."""

    def __init__(self, clock, compile_ms=500.0):
        super().__init__()
        self.net = make_model()
        self.clock = clock
        self.compile_ms = compile_ms
        self.seen = set()
        self.log = []

    @staticmethod
    def run_ms(b):
        return 2.0 + 0.25 * b

    def forward(self, x):
        b = x.shape[0]
        self.log.append({"shape": tuple(x.shape), "training": self.training,
                         "inference": torch.is_inference_mode_enabled(), "grad": torch.is_grad_enabled()})
        ms = self.run_ms(b)
        if b not in self.seen:
            ms += self.compile_ms
            self.seen.add(b)
        self.clock.advance(ms)
        return self.net(x)


def warm_up(model, batch_sizes, clock):
    model.eval()
    took = {}
    with torch.inference_mode():
        for b in batch_sizes:
            t0 = clock()
            model(torch.zeros(b, FEATURES))
            took[b] = clock() - t0
    return took


# ---------------------------------------------------------------- status codes / 状态码
class ServiceState:
    """服务现在的状态。ready：加载和预热做完了没有。predict(rows) -> 概率（list），可能抛异常。
    The service's state. ready: loading and warm-up done? predict(rows) -> probabilities (a list); may raise."""

    def __init__(self, predict, ready=True):
        self.predict = predict
        self.ready = ready


def create_status_app(state):
    app = FastAPI()

    @app.get("/health")
    def health():
        if not state.ready:
            raise HTTPException(status_code=503, detail="loading", headers={"Retry-After": "5"})
        return {"status": "ok"}

    @app.post("/predict")
    def predict(req: PredictRequest):
        if not state.ready:
            raise HTTPException(status_code=503, detail="model is loading", headers={"Retry-After": "5"})
        try:
            probs = state.predict(req.rows)
        except Exception:
            raise HTTPException(status_code=500, detail="internal error")
        return {"probs": probs}

    return app


# ---------------------------------------------------------------- one model, one queue / 一个模型，一个队列
class Ticket:
    """一个请求的「号码牌」。status: None = 还在排队；200 / 500 / 503 / 504 = 有结果了。result: 模型的输出。
    A request's ticket. status: None = still waiting; 200 / 500 / 503 / 504 = done. result: the model's output."""

    def __init__(self, x, arrived):
        self.x = x
        self.arrived = arrived
        self.status = None
        self.result = None

    def __repr__(self):
        return f"Ticket(arrived={self.arrived}, status={self.status})"


class RequestQueue:
    def __init__(self, model_fn, clock, max_waiting=4, timeout_ms=100):
        self.model_fn = model_fn
        self.clock = clock
        self.max_waiting = max_waiting
        self.timeout_ms = timeout_ms
        self.waiting = []

    def submit(self, x):
        t = Ticket(x, self.clock())
        if len(self.waiting) >= self.max_waiting:
            t.status = 503
        else:
            self.waiting.append(t)
        return t

    def run_next(self):
        if not self.waiting:
            return None
        t = self.waiting.pop(0)
        if self.clock() - t.arrived > self.timeout_ms:
            t.status = 504
            return t
        try:
            t.result = self.model_fn(t.x)
            t.status = 200
        except Exception:
            t.status = 500
        return t


# ---------------------------------------------------------------- latency percentiles / 延迟百分位数
def latency_report(times_ms):
    s = sorted(times_ms)
    n = len(s)

    def pick(q):
        k = math.ceil(q * n / 100)
        return s[max(k, 1) - 1]

    return {"p50": pick(50), "p95": pick(95), "p99": pick(99)}


# ---------------------------------------------------------------- micro-batching / 微批处理
class BatchRecorder:
    """把 fn 包一层，记下每一批的大小（sizes）。 Wraps fn and records the size of every batch (sizes)."""

    def __init__(self, fn):
        self.fn = fn
        self.sizes = []

    def __call__(self, xb):
        self.sizes.append(xb.shape[0])
        return self.fn(xb)


class MicroBatcher:
    def __init__(self, run_batch, max_batch, max_wait_ms, clock):
        self.run_batch = run_batch
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.clock = clock
        self.pending = []

    def submit(self, x):
        t = Ticket(x, self.clock())
        self.pending.append(t)
        if len(self.pending) >= self.max_batch:
            self.flush()
        return t

    def poll(self):
        if self.pending and self.clock() - self.pending[0].arrived >= self.max_wait_ms:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        batch, self.pending = self.pending, []
        out = self.run_batch(torch.stack([t.x for t in batch]))
        for t, y in zip(batch, out):
            t.result = y
            t.status = 200
