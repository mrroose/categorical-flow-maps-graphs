# Adapted from https://github.com/NVlabs/edm2/
# and https://github.com/nmboffi/flow-maps/
import math

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


# ----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


# ----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


class MPPositionalEmbedding(nn.Module):
    """
    Deterministic positional embedding with magnitude-preserving scaling.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        half = dim // 2
        assert half % 2 == 0

        # register frequencies as buffer (non-trainable, device-aware)
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: shape (B,) or (B, 1)
        returns:   shape (B, dim)
        """
        t = timesteps.to(torch.float32).view(-1)

        # outer product: (B, half)
        args = torch.outer(t, self.freqs)

        # magnitude-preserving sinusoidal features
        cos_embeddings = torch.cos(args) * math.sqrt(2.0)
        sin_embeddings = torch.sin(args) * math.sqrt(2.0)

        # concatenate along feature dimension
        embedding = torch.cat([cos_embeddings, sin_embeddings], dim=-1)

        return embedding.to(timesteps.dtype)


# ----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


class MPConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: tuple = ()):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel)
        )

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        # if self.training :
        #     # Check if we are inside a torch.func transform (like jvp)
        #     # We use a try/except or a check to avoid in-place mutation errors
        #     # during functional transforms.
        #     try:
        #         with torch.no_grad():
        #             # Attempt forced weight normalization
        #             self.weight.copy_(normalize(w))
        #     except RuntimeError:
        #         # If we are in a jvp/vmap context, self.weight is "captured"
        #         # and cannot be mutated. We simply skip the in-place update.
        #         pass
        w = normalize(w)  # traditional weight normalization

        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


class WeightNetwork(nn.Module):
    """Weight function for heteroscedastic uncertainty weighting."""

    def __init__(self, logvar_channels=128, use_weight=True):
        super().__init__()
        self.use_weight = use_weight
        self.logvar_channels = logvar_channels

        if use_weight:
            # Fourier feature embeddings for s and t
            self.logvar_fourier_s = MPPositionalEmbedding(logvar_channels)
            self.logvar_fourier_t = MPPositionalEmbedding(logvar_channels)

            # Linear projection to scalar log-variance
            self.logvar_linear = MPConv(logvar_channels, 1)

    def forward(self, s, t):
        return self.calc_weight(s, t)

    def calc_weight(self, s, t):
        """
        Calculate learned log-variance weight function.

        Args:
            s: Start time(s), shape (batch_size,) or scalar
            t: End time(s), shape (batch_size,) or scalar

        Returns:
            Log-variance weights, shape (batch_size, 1, 1, 1) for images
        """
        if not self.use_weight:
            return torch.zeros_like(s)

        # Ensure s and t are tensors with batch dimension
        if not isinstance(s, torch.Tensor):
            s = torch.tensor([s], device=next(self.parameters()).device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=next(self.parameters()).device)

        if s.dim() == 0:
            s = s.unsqueeze(0)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Embed s and t using Fourier features
        embed_s = self.logvar_fourier_s(s).view(-1, self.logvar_channels)
        embed_t = self.logvar_fourier_t(t).view(-1, self.logvar_channels)

        # Magnitude-preserving sum (average with scaling)
        embed = mp_sum(embed_s, embed_t, t=0.5)

        # Project to scalar log-variance
        logvar = self.logvar_linear(embed).squeeze(-1)  # [B]

        return logvar
