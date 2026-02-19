import argparse
import logging
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# from torchdiffeq import odeint

logger = logging.getLogger(__name__)


def assert_args(args):
    pass
    # if args.distill_objective != "none":
    #     assert args.distill_objective in {"mse", "csd", "ecld"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def _as_cpu_uint8_tensor(x) -> torch.Tensor:
    """
    torch.set_rng_state expects a CPU uint8 (ByteTensor).
    When checkpoints are loaded with map_location='cuda', the stored RNG state
    may become a CUDA tensor; convert it back safely.
    """
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.device.type != "cpu":
            t = t.to("cpu")
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        return t
    # fallback for older/odd checkpoints (lists/bytes/etc.)
    return torch.as_tensor(x, dtype=torch.uint8, device="cpu")


def _set_rng_state(state_dict: dict):
    random.setstate(state_dict["python"])
    np.random.set_state(state_dict["numpy"])

    if "torch" in state_dict and state_dict["torch"] is not None:
        torch.set_rng_state(_as_cpu_uint8_tensor(state_dict["torch"]))

    # CUDA RNG: try as-is first; if it was mapped to CPU/GPU incorrectly, coerce.
    cuda_state = state_dict.get("cuda", None)
    if torch.cuda.is_available() and cuda_state is not None:
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception:
            try:
                fixed = []
                for i, s in enumerate(cuda_state):
                    t = _as_cpu_uint8_tensor(s).to(f"cuda:{i}")
                    fixed.append(t)
                torch.cuda.set_rng_state_all(fixed)
            except Exception:
                logger.warning(
                    "Could not set CUDA RNG state. Starting with random state."
                )


def _is_slurm():
    return "SLURM_JOB_ID" in os.environ


def get_git_branch():
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_ckpt_name(
    args, name: str | None = None, run_id: str | None = None, epoch: int | None = None
) -> str:
    if name is None:
        legacy_to_objective = {
            "none": "none",
            "mse": "mse",
            "kld": "csd",
            "ecld": "ecld",
        }
        distill_objective = getattr(args, "distill_objective", None)
        if distill_objective is None:
            distill_loss_legacy = getattr(args, "distill_loss", "unknown")
            distill_objective = legacy_to_objective.get(
                distill_loss_legacy, distill_loss_legacy
            )

        parts = [
            f"vel-{args.velocity_loss}",
            f"dist-{distill_objective}",
            f"l{args.num_layers}",
            f"bs{args.batch_size}",
        ]
        if epoch is not None:
            parts.append(f"ep{epoch:04d}")
        if run_id:
            parts.append(f"id-{run_id}")
        else:
            parts.append(datetime.now().strftime("ts-%Y%m%d-%H%M%S"))

        return "ckpt_" + "_".join(parts) + ".pt"
    else:
        return "ckpt_" + name + ".pt"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(args, node_feats, edge_feats):
    from models.transformer import GraphTransformer

    if args.model_size == "tiny":
        hidden_dims = {
            "dx": 16,
            "de": 8,
            "dy": 8,
            "n_head": 2,
            "dim_ffX": 16,
            "dim_ffE": 8,
            "dim_ffy": 8,
        }
        hidden_mlp_dims = {"X": 32, "E": 16, "y": 16}
    elif args.model_size == "floor":
        hidden_dims = {
            "dx": 128,
            "de": 64,
            "dy": 128,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 64,
            "dim_ffy": 256,
        }
        hidden_mlp_dims = {"X": 256, "E": 128, "y": 128}
    elif args.model_size == "abstract":
        hidden_dims = {
            "dx": 256,
            "de": 64,
            "dy": 64,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 64,
            "dim_ffy": 256,
        }
        hidden_mlp_dims = {"X": 128, "E": 64, "y": 128}
    elif args.model_size == "defog":
        # Do not set hidden_mlp_E, dim_ffE too high, computing large tensors on the edges is costly
        hidden_mlp_dims = {"X": 256, "E": 128, "y": 128}

        # The dimensions should satisfy dx % n_head == 0
        hidden_dims = {
            "dx": 256,
            "de": 64,
            "dy": 64,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 128,
            "dim_ffy": 128,
        }
    else:
        raise ValueError("Unknown model_size.")

    model = GraphTransformer(
        input_dims={"X": node_feats, "E": edge_feats, "y": 2},
        hidden_dims=hidden_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        output_dims={"X": node_feats, "E": edge_feats, "y": 2},
        n_layers=args.num_layers,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
        use_time_embed=not args.no_time_embed,
        time_embed_mode=args.tembed_type,
    )

    return model


def get_lambda_d_sched(schedule_type, epochs, lambda_d_start, lambda_d_end):
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    denom = max(1, epochs - 1)

    def frac(epoch):
        # progress in [0, 1]
        e = max(0, min(int(epoch), denom))
        return min(e / denom, 1)

    if schedule_type == "constant":
        return lambda epoch: lambda_d_start

    elif schedule_type == "linear":
        return lambda epoch: lambda_d_start + (lambda_d_end - lambda_d_start) * frac(
            epoch
        )

    elif schedule_type == "cosine":
        import math

        return lambda epoch: lambda_d_end + 0.5 * (lambda_d_start - lambda_d_end) * (
            1 + math.cos(math.pi * frac(epoch))
        )

    else:
        raise ValueError(f"Unknown lambda_d schedule: {schedule_type}")


def _flag_is_set(flag_name: str) -> bool:
    return any(
        arg == flag_name or arg.startswith(f"{flag_name}=") for arg in sys.argv[1:]
    )


def _canonicalize_distillation_flags(parser: argparse.ArgumentParser, args) -> None:
    """
    Canonical internal representation:
      distill_objective in {"none", "mse", "csd", "ecld"}.
    Legacy --distill_loss remains accepted for compatibility.
    """
    legacy_to_objective = {
        "none": "none",
        "mse": "mse",
        "kld": "csd",
        "ecld": "ecld",
    }
    objective_to_legacy = {
        "none": "none",
        "mse": "mse",
        "csd": "kld",
        "ecld": "ecld",
    }

    distill_loss_set = _flag_is_set("--distill_loss")
    objective_set = _flag_is_set("--distill_objective")
    if distill_loss_set:
        mapped_objective = legacy_to_objective[args.distill_loss]
        if objective_set and args.distill_objective != mapped_objective:
            parser.error(
                "Conflicting distillation flags: "
                f"--distill_loss {args.distill_loss} maps to "
                f"--distill_objective {mapped_objective}, but got "
                f"--distill_objective {args.distill_objective}."
            )
        args.distill_objective = mapped_objective

    # Keep legacy field populated to avoid breaking old checkpoint/log consumers.
    args.distill_loss = objective_to_legacy[args.distill_objective]
