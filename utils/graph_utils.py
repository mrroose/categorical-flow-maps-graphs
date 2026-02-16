from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import to_dense_adj, to_dense_batch


def assert_correctly_masked(variable, node_mask):
    assert (
        variable * (1 - node_mask.long())
    ).abs().max().item() < 1e-4, "Variables not masked properly."


def node_mask_to_edge_mask(node_mask: Tensor, diag: Optional[Tensor] = None):
    B, N = node_mask.shape

    if diag is None:
        diag = (
            torch.eye(N, dtype=torch.bool, device=node_mask.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

    node_mask = node_mask.bool()  # [B,N]
    edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2) & (~diag)  # [B,N,N]
    return edge_mask


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:

            # Debug: check non-finites BEFORE masking (helps identify true source)
            if not torch.isfinite(self.E).all():
                n_nan = torch.isnan(self.E).sum().item()
                n_posinf = torch.isposinf(self.E).sum().item()
                n_neginf = torch.isneginf(self.E).sum().item()
                raise AssertionError(
                    "PlaceHolder.mask(): E already has non-finite values BEFORE masking. "
                    f"nan={n_nan}, +inf={n_posinf}, -inf={n_neginf}, "
                    f"total={self.E.numel()}"
                )

            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2

            # Debug: check non-finites AFTER masking (catches inf*0 -> nan issues)
            if not torch.isfinite(self.E).all():
                n_nan = torch.isnan(self.E).sum().item()
                n_posinf = torch.isposinf(self.E).sum().item()
                n_neginf = torch.isneginf(self.E).sum().item()
                raise AssertionError(
                    "PlaceHolder.mask(): E has non-finite values AFTER masking. "
                    f"nan={n_nan}, +inf={n_posinf}, -inf={n_neginf}, "
                    f"total={self.E.numel()} (possible inf*0 -> nan)"
                )

            Et = torch.transpose(self.E, 1, 2)
            if not torch.allclose(self.E, Et, rtol=1e-4, atol=1e-6):
                max_diff = (self.E - Et).abs().max().item()
                raise AssertionError(
                    f"PlaceHolder.mask(): E is not symmetric. max|E-E^T|={max_diff}"
                )

        return self


def to_dense(
    x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor, max_nodes: int
) -> Tuple[PlaceHolder, Tensor, Tensor]:
    """
    Dense batch conversion.

    Notes:
    - Expects edge features where channel 0 corresponds to "no-edge".
    - Produces E of shape [B, N, N, F]. If to_dense_adj returns [B,N,N], it is promoted to [B,N,N,1].
    """
    if edge_index.dtype != torch.long:
        raise TypeError(f"edge_index must be torch.long, got {edge_index.dtype}")

    X_dense, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_nodes)

    # Optional: redundant if you later force diagonal to no-edge anyway.
    # edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)

    E_dense = to_dense_adj(
        edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_nodes
    )

    # Ensure E_dense is [B,N,N,F]
    if E_dense.dim() == 3:
        E_dense = E_dense.unsqueeze(-1)

    # Build "no-edge" vector with matching dtype/device
    no_edge = torch.zeros(E_dense.shape[-1], device=E_dense.device, dtype=E_dense.dtype)
    if no_edge.numel() > 0:
        no_edge[0] = 1

    # Fill missing edges (all-zero feature vector) with no_edge
    mask_no_edge = E_dense.sum(dim=-1) == 0  # [B,N,N]
    E_dense[mask_no_edge] = no_edge

    # Diagonals are set to [0,0]
    diag = (
        torch.eye(E_dense.shape[1], dtype=torch.bool, device=E_dense.device)
        .unsqueeze(0)
        .expand(E_dense.shape[0], -1, -1)
    )
    E_dense[diag] = 0

    return PlaceHolder(X=X_dense, E=E_dense, y=None), node_mask, diag
