"""Day 16 reference: small models and a nested batch shared by the trace_shapes and move_to items
(the practice item and the closed-book item use the same checks)."""
import torch
import torch.nn as nn


class _Twice(nn.Module):
    """enc (a Sequential of two layers) -> act -> head -> act again: the same ReLU runs twice."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 6), nn.Tanh())
        self.act = nn.ReLU()
        self.head = nn.Linear(6, 2)

    def forward(self, x):
        h = self.act(self.enc(x))
        return self.act(self.head(h))


class _Broken(nn.Module):
    """head expects 8 features but body gives 6: forward crashes in head, after body and act ran."""

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.act(self.body(x)))


def _nested_batch():
    """A batch like a DataLoader gives: 6 tensors (float, long, bool) inside dicts, a list and a tuple."""
    g = torch.Generator().manual_seed(0)
    return {
        "input_ids": torch.randint(0, 100, (2, 5), generator=g),
        "pixels": torch.randn(2, 3, 4, 4, generator=g),
        "labels": torch.tensor([1, 0]),
        "extra": {"lengths": [5, 3], "names": ("a", "b"), "mask": torch.ones(2, 5, dtype=torch.bool)},
        "views": [torch.randn(2, 2, generator=g), None, 7],
        "pair": (torch.randn(2, generator=g), "text"),
    }


def _tensors(obj, path="batch"):
    """Yield (path, tensor) for every tensor in a nested dict / list / tuple (used by the checks)."""
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _tensors(v, f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _tensors(v, f"{path}[{i}]")
