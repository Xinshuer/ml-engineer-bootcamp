"""Day 12 参考答案。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TTS_SHAPES = {
    "文本 token id": (4, 20),
    "文本编码器输出 (C=192)": (4, 20, 192),
    "duration 预测 (log 域)": (4, 20),
    "length regulator 之后": (4, 160, 192),
    "解码器输出的 mel": (4, 160, 80),
    "vocoder 输出的波形": (4, 160 * 256),
}


def length_regulator(x, durations, max_len=None):
    outs = [x[b].repeat_interleave(durations[b], dim=0) for b in range(x.size(0))]
    Tm = max_len or max(o.size(0) for o in outs)
    out = x.new_zeros(x.size(0), Tm, x.size(2))
    for b, o in enumerate(outs):
        n = min(o.size(0), Tm)
        out[b, :n] = o[:n]
    return out


def duration_to_boundaries(durations):
    ends = durations.cumsum(0)
    starts = ends - durations
    return torch.stack([starts, ends], dim=-1)


def frame_to_text_index(durations, n_frames):
    idx = torch.arange(durations.numel(), device=durations.device).repeat_interleave(durations)
    return idx[:n_frames]


def make_decoder_inputs(mel):
    go = torch.zeros_like(mel[:, :1])
    return torch.cat([go, mel[:, :-1]], dim=1)


def make_stop_labels(lengths, max_len):
    s = torch.zeros(lengths.numel(), max_len, device=lengths.device)
    s[torch.arange(lengths.numel(), device=lengths.device), lengths - 1] = 1.0
    return s


def masked_mel_loss(pred, target, mask):
    m = mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - target).abs() * m).sum() / (m.sum() * pred.size(-1))


def duration_loss(log_pred, target_dur, mask):
    tgt = torch.log(target_dur.float() + 1)
    m = mask.to(log_pred.dtype)
    return (((log_pred - tgt) ** 2) * m).sum() / m.sum()


def duration_from_log(log_pred):
    return (log_pred.exp() - 1).round().clamp(min=0).long()


def predict_mel_len(log_pred, mask):
    d = duration_from_log(log_pred) * mask.long()
    return d.sum(-1)


def alignment_matrix(durations, n_frames):
    idx = frame_to_text_index(durations, n_frames)
    A = torch.zeros(n_frames, durations.numel(), device=durations.device)
    A[torch.arange(n_frames, device=durations.device), idx] = 1.0
    return A


@torch.no_grad()
def ar_decode(step_fn, n_mels, max_steps, stop_thresh=0.5):
    prev = torch.zeros(1, 1, n_mels)          # go 帧
    frames = []
    for _ in range(max_steps):
        mel_next, stop_logit = step_fn(prev)
        frames.append(mel_next)
        prev = torch.cat([prev, mel_next.unsqueeze(1)], dim=1)
        if torch.sigmoid(stop_logit).item() > stop_thresh:
            break
    return torch.stack(frames, dim=1)


def tts_step_buggy(mel, lengths, log_dur_pred, durations, dur_mask):
    T = mel.size(1)
    dec_in = make_decoder_inputs(mel)                          # 修 1
    mask = (torch.arange(T)[None] < lengths[:, None]).long()
    mel_l = masked_mel_loss(torch.zeros_like(mel), mel, mask)  # 修 2
    dur_l = duration_loss(log_dur_pred, durations, dur_mask)   # 修 3
    return dec_in, mask, mel_l, dur_l
