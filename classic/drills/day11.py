"""Day 11 · 音频的 shape 语言 —— 12 题

运行:  python drills/day11.py
全程只用 torch / torchaudio，不需要 librosa（Python 3.14 上它多半装不上）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check import task, todo, run, true, eq, shape_is, seed, close

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# 01  采样率的算术 —— 音频版的"一个 token 是多少字"
# ==================================================================
def n_samples(seconds, sr):
    """时长 -> 采样点数（取整）。"""
    # --- TODO 01 ---
    raise todo()


def duration(n, sr):
    """采样点数 -> 秒。"""
    # --- TODO 01b ---
    raise todo()


AUDIO_FACTS = {
    "16kHz 下 1 秒有多少个采样点":        None,
    "24kHz 下 2.5 秒有多少个采样点":      None,
    "hop=256 @16kHz 时每秒多少帧":        None,
    "hop=320 @24kHz 时每秒多少帧":        None,
}


@task(1, "采样率算术")
def t01():
    true(n_samples(1.0, 16000) == 16000)
    true(n_samples(0.5, 24000) == 12000)
    true(close(duration(16000, 16000), 1.0))
    want = {
        "16kHz 下 1 秒有多少个采样点": 16000,
        "24kHz 下 2.5 秒有多少个采样点": 60000,
        "hop=256 @16kHz 时每秒多少帧": 62,      # 16000/256 = 62.5 -> 取整 62
        "hop=320 @24kHz 时每秒多少帧": 75,
    }
    wrong = [f"{k}: 你填 {v}, 应为 {want[k]}" for k, v in AUDIO_FACTS.items()
             if v is None or int(v) != want[k]]
    true(not wrong, "\n       " + "\n       ".join(wrong))


# ==================================================================
# 02  STFT 的帧数公式
# ==================================================================
def n_frames(n_sample, n_fft, hop_length, center=True):
    """center=True（默认）时前后各补 n_fft//2，帧数 = n_sample // hop + 1
    center=False 时帧数 = (n_sample - n_fft) // hop + 1
    """
    # --- TODO 02 ---
    raise todo()


@task(2, "STFT 帧数公式")
def t02():
    for n, nf, hop in [(16000, 1024, 256), (12345, 512, 128), (48000, 2048, 512)]:
        real = torch.stft(torch.zeros(n), nf, hop, window=torch.hann_window(nf),
                          return_complex=True, center=True).shape[-1]
        true(n_frames(n, nf, hop, True) == real,
             f"center=True: 你算 {n_frames(n, nf, hop, True)}, 实际 {real}")
        real2 = torch.stft(torch.zeros(n), nf, hop, window=torch.hann_window(nf),
                           return_complex=True, center=False).shape[-1]
        true(n_frames(n, nf, hop, False) == real2,
             f"center=False: 你算 {n_frames(n, nf, hop, False)}, 实际 {real2}")


# ==================================================================
# 03  STFT：从波形到复数谱
# ==================================================================
def stft_mag(wav, n_fft=1024, hop_length=256):
    """wav: (T,) 或 (B, T) -> 幅度谱 (…, n_fft//2 + 1, frames)
    用 torch.hann_window，center=True，return_complex=True，最后取 abs()。
    """
    # --- TODO 03 ---
    raise todo()


@task(3, "STFT 幅度谱")
def t03():
    seed(0)
    w = torch.randn(16000)
    m = stft_mag(w)
    shape_is(m, (513, 63), "1024//2+1 = 513 个频点, 16000//256+1 = 63 帧")
    true((m >= 0).all(), "幅度谱不可能是负的（忘了取 abs？）")
    true(not m.is_complex(), "应该是实数张量")
    shape_is(stft_mag(torch.randn(2, 16000)), (2, 513, 63), "batch 维要能带着走")
    # 一个纯音的能量应该集中在一个频点上
    t = torch.arange(16000).float() / 16000
    tone = torch.sin(2 * math.pi * 1000 * t)
    peak = stft_mag(tone)[:, 30].argmax().item()
    true(abs(peak - 1000 * 1024 / 16000) < 2, f"1000Hz 纯音的峰值频点应该在 64 附近, 实际 {peak}")


# ==================================================================
# 04  mel 滤波器组
# ==================================================================
def hz_to_mel(f):
    """2595 * log10(1 + f / 700)"""
    # --- TODO 04 ---
    raise todo()


def mel_to_hz(m):
    """hz_to_mel 的逆函数。"""
    # --- TODO 04b ---
    raise todo()


@task(4, "mel 刻度换算")
def t04():
    true(close(hz_to_mel(0), 0.0))
    true(close(hz_to_mel(700), 2595 * math.log10(2), 1e-3))
    for f in [100.0, 1000.0, 8000.0]:
        true(close(mel_to_hz(hz_to_mel(f)), f, 1e-2), f"{f}Hz 往返对不上")
    # mel 刻度在低频更"拉伸"
    true(hz_to_mel(200) - hz_to_mel(100) > hz_to_mel(8100) - hz_to_mel(8000),
         "低频 100Hz 的差在 mel 上应该比高频同样的 100Hz 差更大 —— 这就是 mel 的意义")


# ==================================================================
# 05  log-mel：为什么要取 log
# ==================================================================
def to_log_mel(mag, fb, eps=1e-5):
    """mag: (…, n_freq, frames)   fb: (n_mels, n_freq) 的滤波器矩阵
    -> log(fb @ mag + eps)，形状 (…, n_mels, frames)
    """
    # --- TODO 05 ---
    raise todo()


@task(5, "log-mel")
def t05():
    mag = torch.rand(513, 63) * 10
    fb = torch.rand(80, 513)
    lm = to_log_mel(mag, fb)
    shape_is(lm, (80, 63))
    eq(lm, (fb @ mag + 1e-5).log(), tol=1e-4)
    # batch 维
    shape_is(to_log_mel(torch.rand(4, 513, 63), fb), (4, 80, 63), "batch 维要能带着走")
    # log 的意义: 把动态范围压下来
    loud, quiet = torch.full((513, 1), 100.0), torch.full((513, 1), 0.1)
    ratio_lin = (fb @ loud).mean() / (fb @ quiet).mean()
    ratio_log = to_log_mel(loud, fb).mean() / to_log_mel(quiet, fb).mean()
    true(abs(ratio_log) < abs(ratio_lin) / 10, "取 log 之后动态范围应该被大幅压缩")


# ==================================================================
# 06  波形归一化
# ==================================================================
def peak_normalize(wav, target=0.95):
    """把波形按峰值缩放到 target。全零输入要原样返回，不能除以 0。"""
    # --- TODO 06 ---
    raise todo()


@task(6, "峰值归一化")
def t06():
    w = torch.tensor([0.1, -0.4, 0.2])
    o = peak_normalize(w)
    true(close(o.abs().max().item(), 0.95), f"峰值应为 0.95, 实际 {o.abs().max().item():.4f}")
    eq(o / o[0], w / w[0], msg="波形形状不能变，只能整体缩放")
    z = torch.zeros(5)
    eq(peak_normalize(z), z, msg="全零输入不能除以 0 变成 nan")


# ==================================================================
# 07  分帧 —— unfold 的第一次登场
# ==================================================================
def frame(wav, frame_len, hop):
    """wav: (T,) -> (n_frames, frame_len)，相邻帧起点相差 hop，不足的尾巴丢弃。
    用 tensor.unfold，不许写循环。
    """
    # --- TODO 07 ---
    raise todo()


@task(7, "分帧 unfold")
def t07():
    w = torch.arange(10).float()
    f = frame(w, 4, 2)
    shape_is(f, (4, 4))
    eq(f[0], torch.tensor([0., 1., 2., 3.]))
    eq(f[1], torch.tensor([2., 3., 4., 5.]))
    eq(f[3], torch.tensor([6., 7., 8., 9.]))


# ==================================================================
# 08  相位去哪了 —— Griffin-Lim 存在的理由
# ==================================================================
@task(8, "幅度谱丢掉了什么")
def t08():
    seed(1)
    w = torch.randn(4096)
    spec = torch.stft(w, 512, 128, window=torch.hann_window(512), return_complex=True)
    # 用真实相位重建 -> 完美
    perfect = torch.istft(spec, 512, 128, window=torch.hann_window(512), length=4096)
    true((perfect - w).abs().max() < 1e-3, "带相位的重建应该几乎无损")
    # 丢掉相位（全设为 0）再重建 -> 面目全非
    no_phase = torch.istft(spec.abs().to(torch.complex64), 512, 128,
                           window=torch.hann_window(512), length=4096)
    err = (no_phase - w).pow(2).mean() / w.pow(2).mean()
    true(err > 0.5,
         f"丢掉相位后的相对误差只有 {err:.3f}，应该很大。"
         "这就是 vocoder 必须存在的原因：mel 谱里没有相位")


# ==================================================================
# 09  mel 谱就是单通道图像
# ==================================================================
def mel_to_image(mel):
    """mel: (B, n_mels, frames) -> (B, 1, n_mels, frames)
    加一个通道维，之后就能直接喂给 Day 8 写的那些 Conv2d。
    """
    # --- TODO 09 ---
    raise todo()


@task(9, "mel -> conv 输入")
def t09():
    m = torch.randn(4, 80, 100)
    x = mel_to_image(m)
    shape_is(x, (4, 1, 80, 100))
    conv = nn.Conv2d(1, 16, 3, 1, 1)          # Day 8 的积木原样能用
    shape_is(conv(x), (4, 16, 80, 100))


# ==================================================================
# 10  变长音频的 padding 与 mask
# ==================================================================
def pad_mels(mels, pad_value=-11.5):
    """mels: list of (n_mels, T_i)，T 不等长。
    返回 (batch, mask)：
      batch: (B, n_mels, Tmax)，右侧补 pad_value（log-mel 的静音大约是 -11.5）
      mask : (B, Tmax)，有效帧 1，pad 帧 0
    """
    # --- TODO 10 ---
    raise todo()


@task(10, "变长 mel 的 padding")
def t10():
    ms = [torch.ones(4, 3), torch.ones(4, 5) * 2, torch.ones(4, 1) * 3]
    b, mask = pad_mels(ms, pad_value=-11.5)
    shape_is(b, (3, 4, 5))
    shape_is(mask, (3, 5))
    eq(b[0, :, :3], torch.ones(4, 3))
    eq(b[0, :, 3:], torch.full((4, 2), -11.5))
    eq(mask, torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]))


# ==================================================================
# 11  三个张量的 shape 账
# ==================================================================
# 1 秒 16kHz 单声道，n_fft=1024，hop=256，n_mels=80
AUDIO_SHAPES = {
    "waveform":       None,
    "复数谱 (STFT)":   None,
    "幅度谱":          None,
    "log-mel":        None,
    "喂进 Conv2d 前":  None,
}


@task(11, "音频张量 shape 账")
def t11():
    want = {
        "waveform": (16000,),
        "复数谱 (STFT)": (513, 63),
        "幅度谱": (513, 63),
        "log-mel": (80, 63),
        "喂进 Conv2d 前": (1, 1, 80, 63),
    }
    wrong = [f"{k!r}: 你填 {v}, 应为 {want[k]}" for k, v in AUDIO_SHAPES.items()
             if v is None or tuple(v) != want[k]]
    true(not wrong, "\n       " + "\n       ".join(wrong))


# ==================================================================
# 12  找 bug —— 这个 mel 提取有 3 处错
# ==================================================================
def log_mel_buggy(wav, fb, n_fft=1024, hop=256):
    spec = torch.stft(wav, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True)
    mag = spec.real
    mel = fb @ mag
    return torch.log(mel)


@task(12, "找 bug: log_mel_buggy")
def t12():
    seed(2)
    w = torch.randn(8000)
    fb = torch.rand(80, 513)
    got = log_mel_buggy(w, fb)
    want = to_log_mel(stft_mag(w), fb)
    shape_is(got, want.shape)
    true(torch.isfinite(got).all(), "出现了 -inf/nan —— log 里面要加 eps")
    eq(got, want, tol=1e-4,
       msg="三处: 要取 abs() 而不是 real（幅度是复数的模）；log 要加 eps 防止 log(0)；"
           "取完 abs 才能保证非负")


if __name__ == "__main__":
    raise SystemExit(run("Day 11 · 音频基础"))
