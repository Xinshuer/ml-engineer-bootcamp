"""Day 25 参考代码：上线后的监控。
Day 25 reference code: monitoring in production.

题目把这个文件当作隐藏的准备代码加载，然后把题目要你写的名字（targets）删掉。
The exercises load this file as hidden setup, then delete the names you are asked to write (targets).

里面有各题的参考答案，以及检查用的备份（名字以 _ 开头，不会被删掉，也不调用任何会被删掉的名字）。
It holds the reference answers, plus private copies for the checks (names start with _; they are never deleted
and never call a name that may be deleted).
"""
import math
from collections import deque

import torch
import torch.nn as nn


# ---------------------------------------------------------------- 输入漂移：PSI / input drift: PSI
def psi(ref, live, bins=10, eps=1e-4):
    edges = torch.quantile(ref, torch.linspace(0, 1, bins + 1))[1:-1]       # 边界只来自训练数据 / edges from ref only
    e = torch.bincount(torch.bucketize(ref, edges), minlength=bins) / len(ref)
    a = torch.bincount(torch.bucketize(live, edges), minlength=bins) / len(live)
    e, a = e.clamp_min(eps), a.clamp_min(eps)
    return float(((a - e) * torch.log(a / e)).sum())


def fit_reference(train_values, bins=10):
    edges = torch.quantile(train_values, torch.linspace(0, 1, bins + 1))[1:-1]
    shares = torch.bincount(torch.bucketize(train_values, edges), minlength=bins) / len(train_values)
    return {"edges": edges, "shares": shares}


def daily_psi(reference, today_values, eps=1e-4):
    edges, e = reference["edges"], reference["shares"]
    a = torch.bincount(torch.bucketize(today_values, edges), minlength=len(e)) / len(today_values)
    a, e = a.clamp_min(eps), e.clamp_min(eps)
    return float(((a - e) * torch.log(a / e)).sum())


# ---------------------------------------------------------------- KS statistic
def ks_statistic(a, b):
    a = torch.sort(a).values
    b = torch.sort(b).values
    points = torch.cat([a, b])
    cdf_a = torch.searchsorted(a, points, right=True) / len(a)
    cdf_b = torch.searchsorted(b, points, right=True) / len(b)
    return float((cdf_a - cdf_b).abs().max())


# ---------------------------------------------------------------- 预测漂移 / prediction drift
def class_shares(pred, num_classes):
    return torch.bincount(pred, minlength=num_classes) / len(pred)


def prediction_psi(ref_pred, live_pred, num_classes, eps=1e-4):
    e = (torch.bincount(ref_pred, minlength=num_classes) / len(ref_pred)).clamp_min(eps)
    a = (torch.bincount(live_pred, minlength=num_classes) / len(live_pred)).clamp_min(eps)
    return float(((a - e) * torch.log(a / e)).sum())


# ---------------------------------------------------------------- 报警 / alerting
class ErrorRateAlert:
    def __init__(self, window_s=300, threshold=0.05, min_count=50):
        self.window_s = window_s
        self.threshold = threshold
        self.min_count = min_count
        self.events = deque()

    def record(self, t, is_error):
        self.events.append((t, bool(is_error)))

    def firing(self, now):
        while self.events and self.events[0][0] <= now - self.window_s:
            self.events.popleft()
        n = len(self.events)
        if n == 0 or n < self.min_count:
            return False
        errors = sum(1 for _, is_error in self.events if is_error)
        return errors / n > self.threshold


# ---------------------------------------------------------------- 影子流量 / shadow traffic
def serve_with_shadow(x, prod, shadow, stats):
    with torch.inference_mode():
        pred = prod(x).argmax(dim=-1)
        try:
            other = shadow(x).argmax(dim=-1)
        except Exception:
            stats["shadow_errors"] += 1
        else:
            stats["rows"] += len(x)
            stats["disagree"] += int((pred != other).sum())
    return pred


def disagreement_rate(stats):
    if stats["rows"] == 0:
        return None
    return stats["disagree"] / stats["rows"]


# ---------------------------------------------------------------- 金丝雀 / canary
def canary_decision(base_errors, base_n, canary_errors, canary_n, min_n=1000, z=2.0):
    if base_n < min_n or canary_n < min_n:
        return "wait"
    p_base = base_errors / base_n
    p_canary = canary_errors / canary_n
    se = math.sqrt(p_base * (1 - p_base) / base_n + p_canary * (1 - p_canary) / canary_n)
    if p_canary - p_base > z * se:
        return "rollback"
    return "promote"


# ---------------------------------------------------------------- 成本 / cost
def cost_per_1k(gpu_dollars_per_hour, requests_per_second, utilization=1.0):
    requests_per_hour = requests_per_second * 3600 * utilization
    return gpu_dollars_per_hour / requests_per_hour * 1000


# ---------------------------------------------------------------- 检查用 / for the checks
_psi_ref = psi                    # psi 只用 torch，不调用会被删掉的名字 / psi only uses torch
_prediction_psi_ref = prediction_psi


def _ks_slow(a, b):
    """逐点暴力算 KS（纯 Python，O(n²)），和张量版本对拍。 Brute-force KS in plain Python, to compare against."""
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    gaps = [abs(sum(x <= p for x in a) / len(a) - sum(x <= p for x in b) / len(b)) for p in a + b]
    return max(gaps)
