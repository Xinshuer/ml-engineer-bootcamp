"""Day 21 参考代码：导出与压缩（Deployment 1: export and compress）。

题目把这个文件当作隐藏的准备代码加载，然后把题目要你写的名字（targets）删掉。
里面有：今天的小模型 Net（带 BatchNorm 和 Dropout），测试和卡片用的「安静导出」小工具，
以及 model_size_bytes（d21-11 要用）。.onnx 文件只写进 tempfile 建的临时文件夹，用完就删。
"""
import contextlib
import copy
import io
import os
import tempfile
import warnings

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.export / torch.ao 内部的弃用提示，和今天的内容无关。只关掉这两条，
# 「Exporting a model while it is in training mode」这类有用的警告照常显示。
warnings.filterwarnings("ignore", message=r".*LeafSpec.*")
warnings.filterwarnings("ignore", message=r"torch\.ao\.quantization is deprecated")

FEATURES = 8      # 每个样本 8 个特征
CLASSES = 3       # 分 3 类


class Net(nn.Module):
    """8 个特征 -> 3 类的小分类器。BatchNorm 和 Dropout 让 train() / eval() 的行为不一样（故意的）。"""

    def __init__(self, hidden=16, p_drop=0.5):
        super().__init__()
        self.fc1 = nn.Linear(FEATURES, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, CLASSES)

    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.bn(self.fc1(x)))))


def trained_net():
    """一个「训练过」的 Net：BatchNorm 的统计量已经不是初始值。返回时是 train 模式（和刚训练完一样）。"""
    net = Net()
    with torch.no_grad():
        for _ in range(5):
            net(torch.randn(32, FEATURES) * 2 + 1)
    return net


def make_mlp():
    """量化题用的模型：几乎全是 Linear。"""
    return nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))


@contextlib.contextmanager
def quiet():
    """暂时吞掉 stdout / stderr：导出失败时 torch 会打印很长的中间图。"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def short_error(e):
    """一个异常 -> 一行字：最底层原因的类型 + 第一行信息。"""
    root = e
    while root.__cause__ is not None:
        root = root.__cause__
    lines = str(root).strip().splitlines()
    first = lines[0].split(" (unhinted")[0] if lines else ""
    return f"{type(root).__name__}: {first[:160]}"


def export_error(model, *args):
    """用 torch.export.export 试着导出：成功返回 None，失败返回一行原因。"""
    try:
        with quiet():
            torch.export.export(model, args)
        return None
    except Exception as e:  # noqa: BLE001
        return short_error(e)


def onnx_bytes(model, example, dynamic=True):
    """测试用：安静地导出成 ONNX（输入 x，输出 logits，batch 维默认可变），返回文件内容（bytes）。
    文件只在临时文件夹里待一下就删掉。model 的 forward 参数要叫 x。"""
    model.eval()
    kw = dict(dynamo=True, verbose=False, input_names=["x"], output_names=["logits"])
    if dynamic:
        kw["dynamic_shapes"] = {"x": {0: "batch"}}
    with quiet():
        prog = torch.onnx.export(model, (example,), **kw)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        path = os.path.join(folder, "model.onnx")
        prog.save(path)
        with open(path, "rb") as f:
            return f.read()


@contextlib.contextmanager
def onnx_file(data):
    """把 ONNX 的 bytes 写成一个临时 .onnx 文件，with 结束就删掉。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        path = os.path.join(folder, "model.onnx")
        with open(path, "wb") as f:
            f.write(data)
        yield path


def run_onnx(path, x):
    """用 onnxruntime（CPU）跑一个 .onnx 文件：torch 张量进，numpy 数组出。"""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x.detach().cpu().numpy()})[0]


def model_size_bytes(model):
    """state_dict 存下来有多少字节（量化后的权重不是 nn.Parameter，所以不能数 parameters()）。"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes
