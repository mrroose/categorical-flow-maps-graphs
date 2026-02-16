from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _per_graph_loss(values: Tensor, mask: Tensor, *, dims: Tuple[int, ...]) -> Tensor:
    """
    values/mask: same shape starting with batch dim.
    Returns mean over graphs of (sum(values over masked entries) / count(masked entries)).
    """
    mask_f = mask.float()
    num = (values * mask_f).sum(dim=dims)  # Sum over non-padded values
    den = mask_f.sum(dim=dims).clamp(min=1.0)  # Count nr. of non-padded items
    return num / den  # [B]


def instant_velocity_loss(
    model: nn.Module,
    velocity_loss_type: str,
    batch_data: Dict[str, Tensor],
    weight_net: nn.Module,
    label_smoothing: float,
) -> tuple[Tensor, Tensor]:
    x_0 = batch_data["x_0"]
    e_0 = batch_data["e_0"]
    x_1 = batch_data["x_1"]  # [B,N,Fx]
    e_1 = batch_data["e_1"]  # [B,N,N,Fe]
    t = batch_data["t"]  # [B]

    B, N, Fx = x_1.shape
    Fe = e_1.size(-1)
    diag_mask = batch_data["diag_mask"]

    node_mask = batch_data["node_mask"].bool()  # [B,N]
    edge_mask = batch_data["edge_mask"].bool()  # [B,N,N]

    t_x = t.view(B, 1, 1)
    t_e = t.view(B, 1, 1, 1)

    z_t_x = t_x * x_1 + (1.0 - t_x) * x_0
    z_t_e = t_e * e_1 + (1.0 - t_e) * e_0
    z_t_e = 0.5 * (z_t_e + z_t_e.transpose(1, 2))
    z_t_e[diag_mask] = 0.0

    y_t_t = torch.stack((t, t), dim=1)  # [B,2]
    pred_v_t = model(z_t_x.float(), z_t_e.float(), y_t_t.float(), node_mask)

    if velocity_loss_type == "mse":
        target_x = x_1 - x_0
        target_e = e_1 - e_0

        per_node = F.mse_loss(pred_v_t.X, target_x, reduction="none").sum(
            dim=-1
        )  # [B,N]
        per_edge = F.mse_loss(pred_v_t.E, target_e, reduction="none").sum(
            dim=-1
        )  # [B,N,N]

    elif velocity_loss_type == "kld":
        x_target = torch.argmax(x_1, dim=-1)  # [B,N]
        e_target = torch.argmax(e_1, dim=-1)  # [B,N,N]

        per_node = F.cross_entropy(
            pred_v_t.X.reshape(-1, Fx),
            x_target.reshape(-1),
            reduction="none",
            label_smoothing=label_smoothing,
        ).view(B, N)

        per_edge = F.cross_entropy(
            pred_v_t.E.reshape(-1, Fe),
            e_target.reshape(-1),
            reduction="none",
            label_smoothing=label_smoothing,
        ).view(B, N, N)
    else:
        raise ValueError("Unknown velocity loss function.")

    # Compute per-graph losses: [B]
    loss_x_per_graph = _per_graph_loss(per_node, node_mask, dims=(1,))
    loss_e_per_graph = _per_graph_loss(per_edge, edge_mask, dims=(1, 2))

    # Apply heteroscedastic weighting: for diagonal, s = t
    if weight_net is not None and weight_net.use_weight:
        weight_st = weight_net.calc_weight(t, t)  # [B]
        loss_x_per_graph = torch.exp(-weight_st) * loss_x_per_graph + weight_st
        loss_e_per_graph = torch.exp(-weight_st) * loss_e_per_graph + weight_st

    return loss_x_per_graph, loss_e_per_graph


def lagrangian_distill_loss(
    model: nn.Module,
    distill_objective: str,
    batch_data: Dict[str, Tensor],
    weight_net: nn.Module,
    use_scaled_lagrangian: bool = True,
    clamp_min: float = 0.05,
) -> Tuple[Tensor, Tensor]:
    x_0 = batch_data["x_0"]
    e_0 = batch_data["e_0"]
    x_1 = batch_data["x_1"]  # [B,N,Fx]
    e_1 = batch_data["e_1"]  # [B,N,N,Fe]

    s = batch_data["s"]  # [B]
    t = batch_data["t"]  # [B]

    B, N, Fx = batch_data["x_1"].shape
    diag = batch_data["diag_mask"]

    node_mask = batch_data["node_mask"]
    edge_mask = batch_data["edge_mask"]

    device = x_0.device

    s_x = s.view(B, 1, 1)  # for [B,N,Fx]
    s_e = s.view(B, 1, 1, 1)  # for [B,N,N,Fe]
    t_x = t.view(B, 1, 1)  # for [B,N,Fx]
    t_e = t.view(B, 1, 1, 1)  # for [B,N,N,Fe]

    # Construct z_s and z_t (same structure as ballistic.py)
    z_s_x = s_x * x_1 + (1.0 - s_x) * x_0

    z_s_e = s_e * e_1 + (1.0 - s_e) * e_0
    z_s_e = 0.5 * (z_s_e + z_s_e.transpose(1, 2))
    z_s_e[diag] = 0.0

    y_t_t = torch.stack((t, t), dim=1)  # [B,2]
    if distill_objective == "csd":
        denom_s_x = torch.clamp(1.0 - s_x, min=clamp_min)
        denom_s_e = torch.clamp(1.0 - s_e, min=clamp_min)

        dtdt = torch.ones_like(t)

        # Disable autocast for numerical stability
        with torch.amp.autocast(device_type=device.type, enabled=False):

            def _X_st(t_var: torch.Tensor):
                # t_var: [B]
                y_var = torch.stack((s, t_var), dim=1)  # [B,2]
                t_var_x = t_var.view(B, 1, 1)
                t_var_e = t_var.view(B, 1, 1, 1)

                pred_s_t = model(z_s_x.float(), z_s_e.float(), y_var.float(), node_mask)
                mu_st_x = torch.softmax(pred_s_t.X, dim=-1)
                mu_st_e = torch.softmax(pred_s_t.E, dim=-1)

                v_st_x = (mu_st_x - z_s_x) / denom_s_x
                v_st_e = (mu_st_e - z_s_e) / denom_s_e

                out_x = z_s_x + (t_var_x - s_x) * v_st_x
                out_e = z_s_e + (t_var_e - s_e) * v_st_e
                out_e = 0.5 * (out_e + out_e.transpose(1, 2))
                out_e[diag] = 0.0
                return out_x, out_e

            (X_st_x, X_st_e), (dX_dt_x, dX_dt_e) = torch.func.jvp(_X_st, (t,), (dtdt,))

        # sg(v_t(X_s_t))
        with torch.no_grad():
            pred_xt = model(X_st_x.float(), X_st_e.float(), y_t_t.float(), node_mask)
            mu_tt_x = torch.softmax(pred_xt.X, dim=-1)
            mu_tt_e = torch.softmax(pred_xt.E, dim=-1)

            delta_x = mu_tt_x - X_st_x
            delta_e = mu_tt_e - X_st_e
            delta_e = 0.5 * (delta_e + delta_e.transpose(1, 2))
            delta_e[diag] = 0.0

        if use_scaled_lagrangian:
            # time-weighted residual: (1-t) dX/dt - (pi_tt(X)-X)
            res_x = (1.0 - t_x) * dX_dt_x - delta_x
            res_e = (1.0 - t_e) * dX_dt_e - delta_e
        else:
            # unscaled residual: dX/dt - (pi_tt(X)-X)/(1-t)  (clamped)
            denom_t_x = torch.clamp(1.0 - t_x, min=clamp_min)
            denom_t_e = torch.clamp(1.0 - t_e, min=clamp_min)

            v_tt_x = delta_x / denom_t_x
            v_tt_e = delta_e / denom_t_e

            res_x = dX_dt_x - v_tt_x
            res_e = dX_dt_e - v_tt_e

        per_node_sq = (res_x**2).sum(dim=-1)  # [B,N]
        per_edge_sq = (res_e**2).sum(dim=-1)  # [B,N,N]

    elif distill_objective == "mse":
        dtdt = torch.ones_like(t)

        # Disable autocast for numerical stability (mirrors ballistic.py intent)
        with torch.amp.autocast(device_type=device.type, enabled=False):

            def _X_st(t_var: torch.Tensor):
                # t_var: [B]
                y_var = torch.stack((s, t_var), dim=1)  # [B,2]
                t_var_x = t_var.view(B, 1, 1)
                t_var_e = t_var.view(B, 1, 1, 1)

                pred_s_t = model(z_s_x.float(), z_s_e.float(), y_var.float(), node_mask)

                v_st_x = pred_s_t.X
                v_st_e = pred_s_t.E

                out_x = z_s_x + (t_var_x - s_x) * v_st_x
                out_e = z_s_e + (t_var_e - s_e) * v_st_e
                out_e = 0.5 * (out_e + out_e.transpose(1, 2))
                out_e[diag] = 0.0
                return out_x, out_e

            (X_st_x, X_st_e), (dX_dt_x, dX_dt_e) = torch.func.jvp(_X_st, (t,), (dtdt,))

        # sg(v_t(X_s_t))
        with torch.no_grad():
            pred_v_t = model(X_st_x.float(), X_st_e.float(), y_t_t.float(), node_mask)
            v_t_x = pred_v_t.X
            v_t_e = pred_v_t.E
            v_t_e = 0.5 * (v_t_e + v_t_e.transpose(1, 2))
            v_t_e[diag] = 0.0

        per_node_sq = ((dX_dt_x - v_t_x) ** 2).sum(dim=-1)  # [B,N]
        per_edge_sq = ((dX_dt_e - v_t_e) ** 2).sum(dim=-1)  # [B,N,N]

    else:
        raise ValueError("Unknown Lagrangian distillation objective.")

    # Compute per-graph losses: [B]
    loss_x_per_graph = _per_graph_loss(per_node_sq, node_mask.bool(), dims=(1,))
    loss_e_per_graph = _per_graph_loss(per_edge_sq, edge_mask.bool(), dims=(1, 2))

    # Apply heteroscedastic weighting based on (s, t)
    if weight_net is not None and weight_net.use_weight:
        weight_st = weight_net.calc_weight(s, t)  # [B]
        loss_x_per_graph = torch.exp(-weight_st) * loss_x_per_graph + weight_st
        loss_e_per_graph = torch.exp(-weight_st) * loss_e_per_graph + weight_st

    return loss_x_per_graph, loss_e_per_graph


def ecld_loss(
    model: nn.Module,
    batch_data: Dict[str, Tensor],
    weight_net: nn.Module,
    clamp_min: float = 0.05,
) -> Tuple[Tensor, Tensor]:
    x_0 = batch_data["x_0"]
    e_0 = batch_data["e_0"]
    x_1 = batch_data["x_1"]  # [B,N,Fx]
    e_1 = batch_data["e_1"]  # [B,N,N,Fe]

    s = batch_data["s"]  # [B]
    t = batch_data["t"]  # [B]

    B, N, Fx = batch_data["x_1"].shape
    # Fe = batch_data["e_1"].size(-1)
    diag = batch_data["diag_mask"]
    node_mask = batch_data["node_mask"]
    edge_mask = batch_data["edge_mask"]

    device = x_0.device

    s_x = s.view(B, 1, 1)  # for [B,N,Fx]
    s_e = s.view(B, 1, 1, 1)  # for [B,N,N,Fe]
    t_x = t.view(B, 1, 1)  # for [B,N,Fx]
    t_e = t.view(B, 1, 1, 1)  # for [B,N,N,Fe]

    # Construct z_s and z_t (same structure as ballistic.py)
    z_s_x = s_x * x_1 + (1.0 - s_x) * x_0

    z_s_e = s_e * e_1 + (1.0 - s_e) * e_0
    z_s_e = 0.5 * (z_s_e + z_s_e.transpose(1, 2))
    z_s_e[diag] = 0.0

    y_t_t = torch.stack((t, t), dim=1)  # [B,2]

    denom_s_x = torch.clamp(1.0 - s_x, min=clamp_min)
    denom_s_e = torch.clamp(1.0 - s_e, min=clamp_min)

    dtdt = torch.ones_like(t)

    # Disable autocast for numerical stability
    with torch.amp.autocast(device_type=device.type, enabled=False):

        def _mu_st(t_var: torch.Tensor):
            # t_var: [B]
            y_var = torch.stack((s, t_var), dim=1)  # [B,2]
            pred_st = model(z_s_x.float(), z_s_e.float(), y_var.float(), node_mask)
            return pred_st.X, pred_st.E
            # _mu_st_x = torch.softmax(pred_s_t.X, dim=-1)
            # _mu_st_e = torch.softmax(pred_s_t.E, dim=-1)
            # return _mu_st_x, _mu_st_e

        (logits_x, logits_e), (dlogits_dt_x, dlogits_dt_e) = torch.func.jvp(
            _mu_st, (t,), (dtdt,)
        )

        mu_st_x = torch.softmax(logits_x, dim=-1)
        mu_st_e = torch.softmax(logits_e, dim=-1)

        v_st_x = (mu_st_x - z_s_x) / denom_s_x
        v_st_e = (mu_st_e - z_s_e) / denom_s_e

        X_st_x = z_s_x + (t_x - s_x) * v_st_x
        X_st_e = z_s_e + (t_e - s_e) * v_st_e
        X_st_e = 0.5 * (X_st_e + X_st_e.transpose(1, 2))
        X_st_e[diag] = 0.0

    # sg(v_t(X_s_t))
    with torch.no_grad():
        pred_xt = model(X_st_x.float(), X_st_e.float(), y_t_t.float(), node_mask)
        mu_tt_x = torch.softmax(pred_xt.X, dim=-1)
        mu_tt_e = torch.softmax(pred_xt.E, dim=-1)

    logp_X = torch.log_softmax(logits_x, dim=-1)  # [B,N,Fx]
    logp_E = torch.log_softmax(logits_e, dim=-1)  # [B,N,N,Fe]
    L_SD_X = -(mu_tt_x * logp_X).sum(dim=-1)  # [B,N]
    L_SD_E = -(mu_tt_e * logp_E).sum(dim=-1)  # [B,N,N]

    # Chain rule:
    # Jacobian vector product
    # s = softmax(logits), J = diag(s) - ss^T, v = vector
    # Then Jv = s \odot v - s(s^T v)
    # Here, s = mu_st_x, v = dlogits/dt
    # d(softmax)/dt = s * (dlogits/dt - <s, dlogits/dt>)
    # mu_st_x [B, N, Fx], dlogits_dt_x [B, N, Fx]
    # <s, dlogits/dt> = (s * dlogits_dt_x).sum(dim=-1, keepdim=True) (keep last dim)
    inner_x = (mu_st_x * dlogits_dt_x).sum(dim=-1, keepdim=True)
    dmu_st_dt_x = mu_st_x * (dlogits_dt_x - inner_x)
    inner_e = (mu_st_e * dlogits_dt_e).sum(dim=-1, keepdim=True)
    dmu_st_dt_e = mu_st_e * (dlogits_dt_e - inner_e)

    sq_norm_dmu_st_dt_x = (dmu_st_dt_x**2).sum(dim=(-1))  # [B,N]
    sq_norm_dmu_st_dt_e = (dmu_st_dt_e**2).sum(dim=(-1))  # [B,N,N]

    weights_gamma_x = ((t_x - s_x) / denom_s_x).squeeze(-1)
    weights_gamma_e = ((t_e - s_e) / denom_s_e).squeeze(-1)

    L_partial_X = weights_gamma_x * sq_norm_dmu_st_dt_x
    L_partial_E = weights_gamma_e * sq_norm_dmu_st_dt_e

    per_node_sq = 4 * L_SD_X + 2 * L_partial_X
    per_edge_sq = 4 * L_SD_E + 2 * L_partial_E

    # Compute per-graph losses: [B]
    loss_x_per_graph = _per_graph_loss(per_node_sq, node_mask.bool(), dims=(1,))
    loss_e_per_graph = _per_graph_loss(per_edge_sq, edge_mask.bool(), dims=(1, 2))

    # Apply heteroscedastic weighting based on (s, t)
    if weight_net is not None and weight_net.use_weight:
        weight_st = weight_net.calc_weight(s, t)  # [B]
        loss_x_per_graph = torch.exp(-weight_st) * loss_x_per_graph + weight_st
        loss_e_per_graph = torch.exp(-weight_st) * loss_e_per_graph + weight_st

    return loss_x_per_graph, loss_e_per_graph
