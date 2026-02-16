from typing import Dict

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from ema import ExponentialMovingAverage
from loss_functions import (
    ecld_loss,
    instant_velocity_loss,
    lagrangian_distill_loss,
)
from time_sampling import _sample_s_t_diagonal, _sample_s_t_offdiagonal
from utils.graph_utils import node_mask_to_edge_mask, to_dense


def _prepare_dense_batch(batch: Data, *, n_max_nodes: int, device: torch.device):
    """Pure-ish batch prep: densify + masks + diag mask."""
    batch = batch.to(device)

    graph, node_mask, diag_mask = to_dense(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, n_max_nodes
    )
    graph = graph.mask(node_mask)

    x_1, e_1 = graph.X.float(), graph.E.float()  # [B,N,Fx], [B,N,N,Fe]
    node_mask = node_mask.bool()  # [B,N]
    edge_mask = node_mask_to_edge_mask(node_mask, diag_mask)  # [B,N,N]

    return x_1, e_1, node_mask, edge_mask, diag_mask


def _sample_noise_like(x_1: Tensor, e_1: Tensor, diag_mask: Tensor):
    """Sample x0/e0 noise with symmetric edges and zero diagonal."""
    x_0 = torch.randn_like(x_1)
    e_0 = torch.randn_like(e_1)
    e_0 = 0.5 * (e_0 + e_0.transpose(1, 2))
    e_0[diag_mask] = 0.0
    return x_0, e_0


def _split_diag_offdiag(
    batch_size: int,
    *,
    diag_fraction: float,
    distill_objective: str,
    force_diag: bool = False,
    device: torch.device,
):
    """Return diag/offdiag indices; if distillation disabled, all go to diag."""
    perm = torch.randperm(batch_size, device=device)
    if force_diag:
        diag_indices = torch.arange(batch_size, device=device)
        offdiag_bs = (
            int(batch_size * (1 - diag_fraction)) if distill_objective != "none" else 0
        )
        offdiag_indices = perm[:offdiag_bs]
    else:
        diag_bs = (
            int(batch_size * diag_fraction)
            if distill_objective != "none"
            else batch_size
        )
        diag_indices = perm[:diag_bs]
        offdiag_indices = perm[diag_bs:]
    return diag_indices, offdiag_indices


def _maybe_backward_step(
    loss: Tensor,
    *,
    train: bool,
    model: nn.Module,
    weight_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None,
    grad_clip: float,
    step_now: bool,
) -> tuple[dict[str, float], bool, bool]:
    grad_metrics: dict[str, float] = {}
    if not train:
        return grad_metrics, False, False

    loss.backward()

    if not step_now:
        return grad_metrics, True, False

    params = list(model.parameters())
    if weight_net is not None:
        params += list(weight_net.parameters())
    total_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
    grad_metrics["grad_norm_total_clipped"] = float(total_norm.item())
    grad_metrics["grad_clip_val"] = float(grad_clip)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if ema:
        ema.update()

    return grad_metrics, True, True


def train_step(
    train: bool,
    model: nn.Module,
    batch: Data,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    n_max_nodes: int,
    ema: ExponentialMovingAverage | None,
    velocity_loss_type: str,
    distill_objective: str,
    diag_fraction: float,
    force_diag: bool,
    edge_weight: float,
    lambda_distill: float,
    label_smoothing: float,
    clamp_min: float,
    use_scaled_lagrangian: bool,
    weight_net: nn.Module,
    grad_clip: float,
    time_dist: str = "uniform",
    grad_accum_steps: int = 1,
    step_now: bool = True,
) -> tuple[dict[str, float], bool]:
    """
    Run exactly one step on one (PyG) batch.
    Returns metrics normalized per-graph (i.e., divided by batch size B).
    """
    if train:
        model.train()
        assert optimizer is not None, "optimizer must be provided when train=True"
    else:
        model.eval()

    x_1, e_1, node_mask, edge_mask, diag_mask = _prepare_dense_batch(
        batch, n_max_nodes=n_max_nodes, device=device
    )
    batch_size = int(x_1.size(0))

    x_0, e_0 = _sample_noise_like(x_1, e_1, diag_mask)

    diag_indices, offdiag_indices = _split_diag_offdiag(
        batch_size,
        diag_fraction=diag_fraction,
        distill_objective=distill_objective,
        force_diag=force_diag,
        device=device,
    )

    total_loss = torch.zeros((), device=device)
    batch_x_loss = 0.0
    batch_e_loss = 0.0

    metrics: Dict[str, float] = {
        "total_loss": 0.0,
        "x_loss": 0.0,
        "e_loss": 0.0,
        "diag_loss": 0.0,
        "diag_x_loss": 0.0,
        "diag_e_loss": 0.0,
    }

    if distill_objective != "none":
        metrics.update(
            {
                # These will represent the *scaled* (objective) contribution
                "offdiag_loss": 0.0,
                "offdiag_x_loss": 0.0,
                "offdiag_e_loss": 0.0,
                # These represent the *raw* unscaled distillation losses
                "offdiag_loss_raw": 0.0,
                "offdiag_x_loss_raw": 0.0,
                "offdiag_e_loss_raw": 0.0,
            }
        )

    # Diagonal part (s=t)
    diag_count = int(diag_indices.numel())
    if diag_count > 0:
        _, t_diag = _sample_s_t_diagonal(diag_count, device, dist=time_dist)

        batch_data_diag = {
            "x_1": x_1[diag_indices],
            "e_1": e_1[diag_indices],
            "x_0": x_0[diag_indices],
            "e_0": e_0[diag_indices],
            "t": t_diag,
            "node_mask": node_mask[diag_indices],
            "edge_mask": edge_mask[diag_indices],
            "diag_mask": diag_mask[diag_indices],
        }

        loss_x_diag_pg, loss_e_diag_pg = instant_velocity_loss(
            model,
            velocity_loss_type,
            batch_data_diag,
            weight_net,
            label_smoothing,
        )

        loss_x_diag, loss_e_diag = loss_x_diag_pg.mean(), loss_e_diag_pg.mean()
        loss_diag = loss_x_diag + edge_weight * loss_e_diag

        total_loss = total_loss + loss_diag * diag_count
        batch_x_loss += float(loss_x_diag.item()) * diag_count
        batch_e_loss += float(loss_e_diag.item()) * diag_count

        metrics["diag_loss"] += float(loss_diag.item()) * diag_count
        metrics["diag_x_loss"] += float(loss_x_diag.item()) * diag_count
        metrics["diag_e_loss"] += float(loss_e_diag.item()) * diag_count

    # Off-diagonal part (s < t)
    offdiag_count = int(offdiag_indices.numel())
    if offdiag_count > 0 and distill_objective != "none":
        s_offdiag, t_offdiag = _sample_s_t_offdiagonal(
            offdiag_count, device, dist=time_dist
        )

        batch_data_offdiag = {
            "x_1": x_1[offdiag_indices],
            "e_1": e_1[offdiag_indices],
            "x_0": x_0[offdiag_indices],
            "e_0": e_0[offdiag_indices],
            "s": s_offdiag,
            "t": t_offdiag,
            "node_mask": node_mask[offdiag_indices],
            "edge_mask": edge_mask[offdiag_indices],
            "diag_mask": diag_mask[offdiag_indices],
        }

        if distill_objective in {"csd", "mse"}:
            loss_distill_x_pg, loss_distill_e_pg = lagrangian_distill_loss(
                model,
                distill_objective,
                batch_data_offdiag,
                weight_net,
                use_scaled_lagrangian,
                clamp_min,
            )
        elif distill_objective == "ecld":
            loss_distill_x_pg, loss_distill_e_pg = ecld_loss(
                model,
                batch_data_offdiag,
                weight_net,
                clamp_min,
            )
        else:
            raise ValueError(f"Unknown distillation objective: {distill_objective}")

        loss_distill_x, loss_distill_e = (
            loss_distill_x_pg.mean(),
            loss_distill_e_pg.mean(),
        )
        # Raw (unscaled) distillation loss
        loss_offdiag_raw = loss_distill_x + edge_weight * loss_distill_e
        # Scaled (objective) distillation loss
        loss_offdiag = float(lambda_distill) * loss_offdiag_raw

        total_loss = total_loss + loss_offdiag * offdiag_count
        batch_x_loss += float(loss_distill_x.item()) * offdiag_count
        batch_e_loss += float(loss_distill_e.item()) * offdiag_count

        # Scaled metrics (match objective)
        metrics["offdiag_loss"] += float(loss_offdiag.item()) * offdiag_count
        metrics["offdiag_x_loss"] += float(loss_distill_x.item()) * offdiag_count
        metrics["offdiag_e_loss"] += float(loss_distill_e.item()) * offdiag_count

        # Raw metrics (for comparability)
        metrics["offdiag_loss_raw"] += float(loss_offdiag_raw.item()) * offdiag_count
        metrics["offdiag_x_loss_raw"] += float(loss_distill_x.item()) * offdiag_count
        metrics["offdiag_e_loss_raw"] += float(loss_distill_e.item()) * offdiag_count

    loss = total_loss / max(batch_size, 1)
    loss_for_backward = loss
    if train and grad_accum_steps > 1:
        loss_for_backward = loss / float(grad_accum_steps)

    did_backward = False
    if train:
        grad_metrics, did_backward, _ = _maybe_backward_step(
            loss_for_backward,
            train=train,
            model=model,
            weight_net=weight_net,
            optimizer=optimizer,
            ema=ema,
            grad_clip=grad_clip,
            step_now=step_now,
        )

    metrics["total_loss"] = float(loss.item())
    metrics["x_loss"] = batch_x_loss / max(batch_size, 1)
    metrics["e_loss"] = batch_e_loss / max(batch_size, 1)

    # Normalize the rest (they were accumulated as "* bs")
    metrics["diag_loss"] /= max(batch_size, 1)
    metrics["diag_x_loss"] /= max(batch_size, 1)
    metrics["diag_e_loss"] /= max(batch_size, 1)
    if distill_objective != "none":
        metrics["offdiag_loss"] /= max(batch_size, 1)
        metrics["offdiag_x_loss"] /= max(batch_size, 1)
        metrics["offdiag_e_loss"] /= max(batch_size, 1)

        metrics["offdiag_loss_raw"] /= max(batch_size, 1)
        metrics["offdiag_x_loss_raw"] /= max(batch_size, 1)
        metrics["offdiag_e_loss_raw"] /= max(batch_size, 1)

    # Add skip indicator to metrics
    metrics["skipped_backward"] = 0.0 if did_backward else 1.0

    return metrics, did_backward


def run_epoch(
    train: bool,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_max_nodes: int,
    ema,
    velocity_loss_type: str,
    distill_objective: str,
    diag_fraction: float,
    force_diag: bool,
    edge_weight: float,
    weight_net: torch.nn.Module,
    grad_clip: float,
    lambda_distill: float,
    label_smoothing: float,
    clamp_min: float,
    use_scaled_lagrangian: bool,
    time_dist: str = "uniform",
    global_step: int = 0,
    grad_accum_steps: int = 1,
):
    """
    Run one epoch by calling train_step for each batch.
    Aggregates metrics across all steps, weighted by batch size.
    """
    total_graphs = 0
    skipped_batches = 0
    num_batches = len(loader)

    # Initialize metrics accumulator
    metrics_accum = {
        "total_loss": 0.0,
        "x_loss": 0.0,
        "e_loss": 0.0,
        "diag_loss": 0.0,
        "diag_x_loss": 0.0,
        "diag_e_loss": 0.0,
    }
    if distill_objective != "none":
        metrics_accum.update(
            {
                "offdiag_loss": 0.0,
                "offdiag_x_loss": 0.0,
                "offdiag_e_loss": 0.0,
            }
        )

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        # Get batch size from the batch
        batch_size = batch.batch.max().item() + 1 if hasattr(batch, "batch") else 1
        is_last_batch = (batch_idx + 1) == num_batches
        step_now = train and (
            ((batch_idx + 1) % max(grad_accum_steps, 1) == 0) or is_last_batch
        )

        step_metrics, did_backward = train_step(
            train=train,
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            n_max_nodes=n_max_nodes,
            ema=ema,
            velocity_loss_type=velocity_loss_type,
            distill_objective=distill_objective,
            diag_fraction=diag_fraction,
            force_diag=force_diag,
            edge_weight=edge_weight,
            lambda_distill=lambda_distill,
            label_smoothing=label_smoothing,
            clamp_min=clamp_min,
            use_scaled_lagrangian=use_scaled_lagrangian,
            weight_net=weight_net,
            grad_clip=grad_clip,
            time_dist=time_dist,
            grad_accum_steps=grad_accum_steps,
            step_now=step_now,
        )

        if not did_backward and train:
            skipped_batches += 1

        # Only accumulate metrics from non-skipped batches
        if did_backward or not train:
            for k in metrics_accum:
                if k in step_metrics:
                    metrics_accum[k] += step_metrics[k] * batch_size
            total_graphs += batch_size

        global_step += 1

    # Normalize by total graphs
    total_graphs = max(total_graphs, 1)
    metrics = {k: v / total_graphs for k, v in metrics_accum.items()}
    metrics["skipped_batches"] = skipped_batches

    if not train:
        global_step = None  # Ensure we don't accidentally update it during validation

    return model, metrics, global_step
