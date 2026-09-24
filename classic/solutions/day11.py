"""Day 11 参考答案。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

AUDIO_FACTS = {
    "16kHz 下 1 秒有多少个采样点": 16000,
    "24kHz 下 2.5 秒有多少个采样点": 60000,
    "hop=256 @16kHz 时每秒多少帧": 62,
    "hop=320 @24kHz 时每秒多少帧": 75,
}

AUDIO_SHAPES = {
    "waveform": (16000,),
    "复数谱 (STFT)": (513, 63),
    "幅度谱": (513, 63),
    "log-mel": (80, 63),
    "喂进 Conv2d 前": (1, 1, 80, 63),
}


def n_samples(seconds, sr):
    return int(seconds * sr)


def duration(n, sr):
    return n / sr


def n_frames(n_sample, n_fft, hop_length, center=True):
    if center:
        return n_sample // hop_length + 1
    return (n_sample - n_fft) // hop_length + 1


def stft_mag(wav, n_fft=1024, hop_length=256):
    win = torch.hann_window(n_fft, device=wav.device)
    return torch.stft(wav, n_fft, hop_length, window=win,
                      center=True, return_complex=True).abs()


def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def to_log_mel(mag, fb, eps=1e-5):
    return torch.log(fb @ mag + eps)


def peak_normalize(wav, target=0.95):
    peak = wav.abs().max()
    if peak == 0:
        return wav
    return wav * (target / peak)


def frame(wav, frame_len, hop):
    return wav.unfold(0, frame_len, hop)


def mel_to_image(mel):
    return mel.unsqueeze(1)


def pad_mels(mels, pad_value=-11.5):
    B = len(mels)
    n_mels = mels[0].size(0)
    Tmax = max(m.size(1) for m in mels)
    batch = torch.full((B, n_mels, Tmax), pad_value)
    mask = torch.zeros(B, Tmax, dtype=torch.long)
    for i, m in enumerate(mels):
        T = m.size(1)
        batch[i, :, :T] = m
        mask[i, :T] = 1
    return batch, mask


def log_mel_buggy(wav, fb, n_fft=1024, hop=256):
    spec = torch.stft(wav, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True)
    mag = spec.abs()                    # 修 1+3: real -> abs，保证非负
    mel = fb @ mag
    return torch.log(mel + 1e-5)        # 修 2: 加 eps
