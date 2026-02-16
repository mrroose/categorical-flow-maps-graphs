import torch
import torch.nn as nn

from loss_weights import MPConv, MPPositionalEmbedding, mp_silu, mp_sum


class YEncoder(nn.Module):
    def __init__(
        self,
        mp_time_dim: int,
        out_dim: int,
        hidden_dim: int,
        t_minus_s: bool,
        mode: str = "concat",  # concat or sum
    ):
        super().__init__()

        self.mode = mode
        self.t_minus_s = t_minus_s

        # Both modes use the same base Fourier embeddings
        self.emb_s_fourier = MPPositionalEmbedding(mp_time_dim)
        self.emb_t_fourier = MPPositionalEmbedding(mp_time_dim)

        if mode == "sum":
            # Boffi style: Individual projections before summation
            self.proj_s = MPConv(mp_time_dim, out_dim)
            self.proj_t = MPConv(mp_time_dim, out_dim)
        elif mode == "concat":
            self.mlp = nn.Sequential(
                nn.Linear(2 * mp_time_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, y):
        # y is [B, 2] -> (s, t)
        s_raw = y[:, 0]
        t_raw = y[:, 1]

        # 1. Apply t-s logic if requested
        # Usually 'sum' uses (s, t-s) and 'concat' uses (s, t)
        a_val = s_raw
        b_val = (t_raw - s_raw) if self.t_minus_s else t_raw

        # 2. Fourier Embeddings
        emb_a = self.emb_s_fourier(a_val)
        emb_b = self.emb_t_fourier(b_val)

        if self.mode == "sum":
            h_a = self.proj_s(emb_a)
            h_b = self.proj_t(emb_b)

            # mp_sum keeps variance = 1.0
            h = mp_sum(h_a, h_b, t=0.5)
            # mp_silu provides gain-corrected activation
            return mp_silu(h)
        elif self.mode == "concat":
            h = torch.cat([emb_a, emb_b], dim=-1)
            return self.mlp(h)
        else:
            raise ValueError("This should not be reached.")
