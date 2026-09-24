# imports of the drill file (the checks use them)
import sys, os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Day 07 参考答案。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_SHAPES = {
    (32, 3, 1, 1): 32,
    (32, 4, 2, 1): 16,
    (32, 3, 2, 1): 16,
    (32, 1, 1, 0): 32,
    (7, 3, 2, 0): 3,
}


def conv_out(H, kernel, stride, padding):
    return (H + 2 * padding - kernel) // stride + 1


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def kl_normal(mu, logvar):
    return 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).flatten(1).sum(1)


class Encoder(nn.Module):
    def __init__(self, z_ch=4):
        super().__init__()
        self.z_ch = z_ch
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 2 * z_ch, 3, 1, 1),
        )

    def forward(self, x):
        h = self.net(x)
        return h[:, :self.z_ch], h[:, self.z_ch:]


class Decoder(nn.Module):
    def __init__(self, z_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_ch, 64, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = ((x_hat - x) ** 2).flatten(1).sum(1).mean()
    kl = kl_normal(mu, logvar).mean()
    return recon + beta * kl, recon, kl


def compression(H, W, C_img, h, w, C_z):
    return (H * W * C_img) / (h * w * C_z)


def fit_scaling_factor(latents):
    return 1.0 / latents.std().item()


def collapsed_dims(mu, logvar, thresh=0.01):
    per_dim = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)   # (B, D)
    return int((per_dim.mean(0) < thresh).sum())


def vae_step_buggy(x, x_hat, mu, logvar, beta=1.0):
    std = torch.exp(0.5 * logvar)                                       # 修 1
    z = mu + std * torch.randn_like(std)
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl = 0.5 * (mu ** 2 + torch.exp(logvar) - 1 - logvar).sum() / x.size(0)   # 修 2
    return recon + beta * kl, z                                         # 修 3
