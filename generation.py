from typing import List

import torch
from torch import nn as nn
from torch.nn import functional as F

from utils.graph_utils import PlaceHolder, node_mask_to_edge_mask
from utils.molecule_utils import make_molecule


def generate_graphs_euler(
    model: nn.Module,
    num_steps: int,
    num_mols: int,
    max_nodes: int,
    node_feats: int,
    edge_feats: int,
    device: torch.device,
    velocity_loss: str,
    counter: torch.Tensor,
    batch_size: int,
    clamp_min: float,
    argmax: bool = True,
) -> List:
    N = max_nodes
    Fx = node_feats
    Fe = edge_feats

    time_steps = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device)
    step_size = 1.0 / num_steps

    # Default batch size equals num_mols if not provided or invalid
    B_total = int(num_mols)
    if batch_size is None or batch_size <= 0:
        batch_size = B_total

    all_mols: List = []

    p_n_nodes = (counter / counter.sum()).to(device)

    model.eval()
    with torch.no_grad():
        for start in range(0, B_total, batch_size):
            B = min(batch_size, B_total - start)

            # Initialize noisy state per batch
            x_t = torch.randn(B, N, Fx, device=device)
            e_t = torch.randn(B, N, N, Fe, device=device)
            e_t = (e_t + torch.transpose(e_t, 1, 2)) / 2

            # Sample number of nodes per graph
            n_nodes_per_graph = torch.multinomial(p_n_nodes, B, replacement=True)
            node_mask = torch.stack(
                [
                    torch.cat(
                        [
                            torch.ones(sample, dtype=torch.bool, device=device),
                            torch.zeros(N - sample, dtype=torch.bool, device=device),
                        ]
                    )
                    for sample in n_nodes_per_graph
                ],
                dim=0,
            )

            graph = PlaceHolder(X=x_t, E=e_t, y=None).mask(node_mask)
            x_t, e_t = graph.X.float(), graph.E.float()

            diag = (
                torch.eye(N, dtype=torch.bool, device=device)
                .unsqueeze(0)
                .expand(B, -1, -1)
            )

            # Euler rollout
            for i in range(num_steps):
                t = time_steps[i]
                y_t = torch.full((B, 2), fill_value=t.item(), device=device)

                pred = model(x_t, e_t, y_t, node_mask)

                if velocity_loss == "mse":
                    v_x, v_e = pred.X, pred.E
                elif velocity_loss == "kld":
                    mu_x = torch.softmax(pred.X, dim=-1)
                    mu_e = torch.softmax(pred.E, dim=-1)
                    denom = torch.clamp(1.0 - t, min=clamp_min)
                    v_x = (mu_x - x_t) / denom
                    v_e = (mu_e - e_t) / denom
                else:
                    raise ValueError("Unknown velocity loss function.")

                x_t = x_t + step_size * v_x
                e_t = e_t + step_size * v_e

                e_t = 0.5 * (e_t + e_t.transpose(1, 2))
                e_t[diag] = 0.0

                graph = PlaceHolder(X=x_t, E=e_t, y=None).mask(node_mask)
                x_t, e_t = graph.X.float(), graph.E.float()

            # Discretize to valid one-hot
            if argmax:
                x_idx = torch.argmax(x_t, dim=-1)
                e_idx = torch.argmax(e_t, dim=-1)
            else:
                x_idx = torch.multinomial(
                    x_t.reshape(-1, x_t.size(-1)), num_samples=1
                ).squeeze(-1)
                x_idx = x_idx.view(*x_t.shape[:-1])

                e_idx = torch.multinomial(
                    e_t.reshape(-1, e_t.size(-1)), num_samples=1
                ).squeeze(-1)
                e_idx = e_idx.view(*e_t.shape[:-1])
            x_t = F.one_hot(x_idx, num_classes=Fx).to(x_t.dtype)
            x_t = x_t * node_mask.unsqueeze(-1).to(x_t.dtype)

            triu = torch.triu(
                torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1
            ).unsqueeze(0)
            e_idx = torch.where(triu, e_idx, e_idx.transpose(1, 2))
            e_idx = e_idx.masked_fill(diag, 0)

            e_t = F.one_hot(e_idx, num_classes=Fe).to(e_t.dtype)
            edge_mask = node_mask_to_edge_mask(node_mask, diag)
            e_t = e_t * edge_mask.unsqueeze(-1).to(e_t.dtype)

            batch_mols = [
                make_molecule(x, e, int(node_mask[i].sum().item()))
                for i, (x, e) in enumerate(zip(x_t, e_t))
            ]
            all_mols.extend(batch_mols)

    return all_mols[:B_total]


def generate_graphs_flow_map(
    model: nn.Module,
    num_steps: int,
    num_mols: int,
    max_nodes: int,
    node_feats: int,
    edge_feats: int,
    device: torch.device,
    velocity_loss: str,
    counter: torch.Tensor,
    batch_size: int,
    clamp_min: float,
    argmax: bool = True,
) -> List:
    """
    Flow-map-style multi-step sampler for graphs with batch-level sampling:
    - Uses two different timesteps (s,t) per step.
    - Velocity types:
        * "mse": use model outputs directly as velocity.
        * "kld": use (softmax(pred) - z) / (1 - s) as velocity.
    """
    N = max_nodes
    Fx = node_feats
    Fe = edge_feats

    time_steps = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device)

    # Default batch size equals num_mols if not provided or invalid
    B_total = int(num_mols)
    if batch_size is None or batch_size <= 0:
        batch_size = B_total

    all_mols: List = []

    # Sample graph sizes from empirical counter
    p_n_nodes = (counter / counter.sum()).to(device)

    model.eval()
    with torch.no_grad():
        for start in range(0, B_total, batch_size):
            B = min(batch_size, B_total - start)

            # Initialize noisy state per batch
            x_t = torch.randn(B, N, Fx, device=device)
            e_t = torch.randn(B, N, N, Fe, device=device)
            e_t = 0.5 * (e_t + e_t.transpose(1, 2))  # symmetrize

            # Sample number of nodes per graph
            n_nodes_per_graph = torch.multinomial(p_n_nodes, B, replacement=True)
            node_mask = torch.stack(
                [
                    torch.cat(
                        [
                            torch.ones(sample, dtype=torch.bool, device=device),
                            torch.zeros(N - sample, dtype=torch.bool, device=device),
                        ]
                    )
                    for sample in n_nodes_per_graph
                ],
                dim=0,
            )

            graph = PlaceHolder(X=x_t, E=e_t, y=None).mask(node_mask)
            x_t, e_t = graph.X.float(), graph.E.float()

            diag = (
                torch.eye(N, dtype=torch.bool, device=device)
                .unsqueeze(0)
                .expand(B, -1, -1)
            )

            # Rollout with (s, t) per step
            for i in range(num_steps):
                s = time_steps[i]
                t = time_steps[i + 1]
                dt = (t - s).item()

                # y = (s, t) for each graph in the batch
                y_st = torch.stack(
                    (
                        torch.full((B,), s.item(), device=device),
                        torch.full((B,), t.item(), device=device),
                    ),
                    dim=1,
                )

                # Predict velocity at (s,t)
                pred = model(x_t, e_t, y_st, node_mask)

                if velocity_loss == "mse":
                    v_x, v_e = pred.X, pred.E
                elif velocity_loss == "kld":
                    mu_x = torch.softmax(pred.X, dim=-1)
                    mu_e = torch.softmax(pred.E, dim=-1)
                    denom_x = torch.clamp(1.0 - s, min=clamp_min).view(1, 1, 1)
                    denom_e = torch.clamp(1.0 - s, min=clamp_min).view(1, 1, 1, 1)
                    v_x = (mu_x - x_t) / denom_x
                    v_e = (mu_e - e_t) / denom_e
                else:
                    raise ValueError("Unknown velocity loss function.")

                # Integrate from s to t
                x_t = x_t + dt * v_x
                e_t = e_t + dt * v_e

                # Enforce symmetry and zero diagonal
                e_t = 0.5 * (e_t + e_t.transpose(1, 2))
                e_t[diag] = 0.0

                # Re-apply masking
                graph = PlaceHolder(X=x_t, E=e_t, y=None).mask(node_mask)
                x_t, e_t = graph.X.float(), graph.E.float()

            # Discretize to valid one-hot graph
            if argmax:
                x_idx = torch.argmax(x_t, dim=-1)
                e_idx = torch.argmax(e_t, dim=-1)
            else:
                x_idx = torch.multinomial(
                    x_t.reshape(-1, x_t.size(-1)), num_samples=1
                ).squeeze(-1)
                x_idx = x_idx.view(*x_t.shape[:-1])

                e_idx = torch.multinomial(
                    e_t.reshape(-1, e_t.size(-1)), num_samples=1
                ).squeeze(-1)
                e_idx = e_idx.view(*e_t.shape[:-1])

            x_t = F.one_hot(x_idx, num_classes=Fx).to(x_t.dtype)
            x_t = x_t * node_mask.unsqueeze(-1).to(x_t.dtype)

            triu = torch.triu(
                torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1
            ).unsqueeze(0)
            e_idx = torch.where(triu, e_idx, e_idx.transpose(1, 2))
            e_idx = e_idx.masked_fill(diag, 0)

            e_t = F.one_hot(e_idx, num_classes=Fe).to(e_t.dtype)
            edge_mask = node_mask_to_edge_mask(node_mask, diag)
            e_t = e_t * edge_mask.unsqueeze(-1).to(e_t.dtype)

            batch_mols = [
                make_molecule(x, e, int(node_mask[i].sum().item()))
                for i, (x, e) in enumerate(zip(x_t, e_t))
            ]
            all_mols.extend(batch_mols)

    return all_mols[:B_total]
