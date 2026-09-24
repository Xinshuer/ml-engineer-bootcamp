"""Day 26 reference: the capstone pipeline, one function per step.

Every item loads this whole file as hidden setup and then deletes the names the learner writes
(`targets:`), so the learner's version replaces the reference one and the later steps call it.

    step 1  data + checks     make_data, TinyNet, smoke_test, overfit_one_batch
    step 2  train             train_steps, accuracy, train_model
    step 3  checkpoint        save_checkpoint, load_checkpoint
    step 4  export            export_onnx, verify_onnx
    step 5  service           Row, PredictRequest, torch_predictor, onnx_predictor, create_app
    step 6  monitoring        measure_latency, latency_report, psi
"""
import math
import os
import time
import warnings
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel, Field, FiniteFloat
from safetensors.torch import load_file, save_file

# noise from the installed library versions (starlette's TestClient, the ONNX exporter); harmless here
warnings.filterwarnings("ignore", message=".*httpx.*")
warnings.filterwarnings("ignore", message=".*LeafSpec.*")

FEATURES = 8                                  # 8 sensor readings per machine
CLASSES = 3
LABELS = ["ok", "overheat", "vibration"]      # what the model predicts
MAX_ROWS = 64                                 # the most rows one request may carry


# ---------------------------------------------------------------- step 1: data and checks
def class_centers():
    """(3, 8): the mean sensor readings of each class. Fixed forever (its own seed)."""
    g = torch.Generator().manual_seed(1234)
    return torch.randn(CLASSES, FEATURES, generator=g)


def make_data(n, seed=0, shift=0.0):
    """n machines -> x (n, 8) float32, y (n,) int64. Same seed -> same data; the global RNG is not touched."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, CLASSES, (n,), generator=g)
    x = class_centers()[y] + torch.randn(n, FEATURES, generator=g) + shift
    return x, y


class TinyNet(nn.Module):
    """8 readings -> 3 logits. Dropout makes train() and eval() behave differently (on purpose)."""

    def __init__(self, hidden=32, p_drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(FEATURES, hidden)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, CLASSES)

    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))


def smoke_test(model, x, y):
    """One forward + backward before any long run. Returns the first loss (float); raises AssertionError."""
    logits = model(x)
    if tuple(logits.shape) != (len(x), CLASSES):
        raise AssertionError(f"logits shape {tuple(logits.shape)}, expected {(len(x), CLASSES)}")
    if not torch.isfinite(logits).all():
        raise AssertionError("logits contain nan / inf")
    loss = F.cross_entropy(logits, y)
    if abs(loss.item() - math.log(CLASSES)) > 0.5:
        raise AssertionError(f"first loss {loss.item():.3f} is far from ln {CLASSES} = {math.log(CLASSES):.3f}")
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
            raise AssertionError(f"{name} got no (finite) gradient")
    return loss.item()


def overfit_one_batch(model, x, y, steps=200, lr=1e-2):
    """Train on the same small batch again and again; the loss must get close to 0."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------- step 2: train
def train_steps(model, opt, x, y, steps, batch_size=64):
    """`steps` optimisation steps on random mini-batches drawn with the global RNG. Returns the losses."""
    model.train()
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch_size,))
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def accuracy(model, x, y):
    """Share of rows whose top class is right. Leaves the model in eval mode."""
    model.eval()
    with torch.inference_mode():
        return (model(x).argmax(dim=1) == y).float().mean().item()


def train_model(steps=300, seed=0, lr=3e-3):
    """The whole of step 2 in one call: returns (model, opt, losses). Well under a second on a CPU."""
    x, y = make_data(2000, seed=1)
    torch.manual_seed(seed)
    model = TinyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = train_steps(model, opt, x, y, steps)
    return model, opt, losses


# ---------------------------------------------------------------- step 3: checkpoint
def save_checkpoint(folder, model, opt, step):
    """folder/model.safetensors (weights only) + folder/train_state.pt (optimizer, step, RNG state)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), folder / "model.safetensors")
    torch.save({"opt": opt.state_dict(), "step": step, "rng": torch.get_rng_state()}, folder / "train_state.pt")


def load_checkpoint(folder, model, opt):
    """Restore everything save_checkpoint wrote; returns the step to continue from."""
    folder = Path(folder)
    model.load_state_dict(load_file(folder / "model.safetensors"))
    state = torch.load(folder / "train_state.pt", weights_only=True)
    opt.load_state_dict(state["opt"])
    torch.set_rng_state(state["rng"])
    return state["step"]


# ---------------------------------------------------------------- step 4: export
def export_onnx(model, path):
    """model -> one ONNX file with a dynamic batch dimension. Returns the path as a string."""
    model.eval()
    example = torch.randn(2, FEATURES)
    torch.onnx.export(model, (example,), str(path), dynamo=True,
                      input_names=["x"], output_names=["logits"],
                      dynamic_shapes={"x": {0: "batch"}},
                      external_data=False, verbose=False)
    return str(path)


def verify_onnx(model, path, batch_sizes=(1, 5, 64), tol=1e-4):
    """Run the ONNX file (onnxruntime) and the model (PyTorch) on the same inputs; return the max |difference|."""
    import onnxruntime as ort

    model.eval()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    worst = 0.0
    for b in batch_sizes:
        x = torch.randn(b, FEATURES)
        with torch.inference_mode():
            want = model(x).numpy()
        got = sess.run(None, {name: x.numpy()})[0]
        worst = max(worst, float(np.abs(got - want).max()))
    if worst > tol:
        raise AssertionError(f"ONNX and PyTorch differ by {worst:.2e} > {tol}")
    return worst


# ---------------------------------------------------------------- step 5: service
Row = Annotated[list[FiniteFloat], Field(min_length=FEATURES, max_length=FEATURES)]


class PredictRequest(BaseModel):
    rows: list[Row] = Field(min_length=1, max_length=MAX_ROWS)


def torch_predictor(model):
    """A predict function backed by PyTorch: np.ndarray (B, 8) -> probabilities (B, 3)."""
    model.eval()

    def predict(x):
        with torch.inference_mode():
            logits = model(torch.as_tensor(np.asarray(x, dtype=np.float32)))
            return torch.softmax(logits, dim=-1).numpy()

    return predict


def onnx_predictor(path):
    """The same interface, backed by onnxruntime (what the real service loads)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def predict(x):
        logits = sess.run(None, {name: np.asarray(x, dtype=np.float32)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    return predict


def create_app(load_model):
    """load_model() -> predict function. Load once, warm up once, then serve."""
    predict = load_model()
    predict(np.zeros((1, FEATURES), dtype=np.float32))
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict_route(req: PredictRequest):
        x = np.asarray(req.rows, dtype=np.float32)
        probs = predict(x)
        return {"labels": [LABELS[i] for i in probs.argmax(axis=1)], "probs": probs.tolist()}

    return app


# ---------------------------------------------------------------- step 6: monitoring
def measure_latency(client, payload, n=50, warmup=5):
    """POST payload to /predict warmup + n times; return the n latencies in milliseconds."""
    for _ in range(warmup):
        client.post("/predict", json=payload)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post("/predict", json=payload)
        times.append((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            raise AssertionError(f"status {r.status_code}: only successful requests count")
    return times


def latency_report(times_ms):
    t = np.asarray(times_ms, dtype=float)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99))}


def psi(expected, actual, bins=10, eps=1e-4):
    """Population stability index of one feature: training values (expected) vs live values (actual)."""
    expected = torch.as_tensor(expected, dtype=torch.float64).flatten()
    actual = torch.as_tensor(actual, dtype=torch.float64).flatten()
    edges = torch.quantile(expected, torch.linspace(0, 1, bins + 1, dtype=torch.float64))[1:-1]
    e = torch.bincount(torch.bucketize(expected, edges), minlength=bins) / len(expected)
    a = torch.bincount(torch.bucketize(actual, edges), minlength=bins) / len(actual)
    e, a = e.clamp_min(eps), a.clamp_min(eps)
    return float(((a - e) * torch.log(a / e)).sum())
