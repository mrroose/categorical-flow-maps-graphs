import torch

_DEFAULT_P_MEAN = -0.4
_DEFAULT_P_STD = 1.0


def _sample_t(num_samples: int, device: torch.device, dist: str) -> torch.Tensor:
    """
    Sample timesteps in [0,1].
    Supported dist:
      - "uniform"
      - "logit_normal" (aka "log_normal" for backward compatibility)
    """
    dist = "logit_normal" if dist == "log_normal" else dist
    return _timestep_sample(
        _DEFAULT_P_MEAN, _DEFAULT_P_STD, num_samples, device, dist=dist
    )


def _sample_s_t_diagonal(B: int, device: torch.device, dist: str = "uniform"):
    """Sample s=t for diagonal term."""
    t = _sample_t(B, device, dist)
    return t, t


def _sample_s_t_offdiagonal(B: int, device: torch.device, dist: str = "uniform"):
    """Sample s < t for off-diagonal term."""
    a = _sample_t(B, device, dist)
    b = _sample_t(B, device, dist)
    s = torch.minimum(a, b)
    t = torch.maximum(a, b)
    return s, t


def logit_normal_timestep_sample(
    P_mean: float, P_std: float, num_samples: int, device: torch.device
) -> torch.Tensor:
    rnd_normal = torch.randn((num_samples,), device=device)
    time = torch.sigmoid(rnd_normal * P_std + P_mean)
    time = torch.clip(time, min=0.0, max=1.0)
    return time


def uniform_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    rnd = torch.rand((num_samples,), device=device)
    time = torch.clip(rnd, min=0.0, max=1.0)
    return time


def _timestep_sample(
    P_mean: float,
    P_std: float,
    num_samples: int,
    device: torch.device,
    dist: str,
) -> torch.Tensor:
    if dist in ("logit_normal", "log_normal"):
        return logit_normal_timestep_sample(P_mean, P_std, num_samples, device=device)
    if dist == "uniform":
        return uniform_timestep_sample(num_samples, device=device)
    raise ValueError(f"Unknown timestep distribution: {dist}")


def sample_two_timesteps(
    num_samples: int,
    device: torch.device,
    tr_sampler,
    P_mean_t,
    P_std_t,
    P_mean_r,
    P_std_r,
    ratio,
    dist,
):
    if tr_sampler == "v0":
        t, r = sample_two_timesteps_t_r_v0(
            P_mean_t,
            P_std_t,
            P_mean_r,
            P_std_r,
            ratio,
            num_samples,
            device=device,
            dist=dist,
        )
        return t, r
    elif tr_sampler == "v1":
        t, r = sample_two_timesteps_t_r_v1(
            P_mean_t,
            P_std_t,
            P_mean_r,
            P_std_r,
            ratio,
            num_samples,
            device=device,
            dist=dist,
        )
        return t, r
    else:
        raise ValueError(f"Unknown joint time sampler: {tr_sampler}")


def sample_two_timesteps_t_r_v0(
    P_mean_t,
    P_std_t,
    P_mean_r,
    P_std_r,
    ratio,
    num_samples: int,
    device: torch.device,
    dist: str,
):
    """
    Sampler (t, r): independently sample t and r, with post-processing.
    Version 0: used in paper.
    """
    # step 1: sample two independent timesteps
    t = _timestep_sample(P_mean_t, P_std_t, num_samples, device=device, dist=dist)
    r = _timestep_sample(P_mean_r, P_std_r, num_samples, device=device, dist=dist)

    # step 2: ensure t >= r
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    # step 3: make t and r different with a probability of args.ratio
    prob = torch.rand(num_samples, device=device)
    mask = prob < 1 - ratio
    r = torch.where(mask, t, r)

    return t, r


def sample_two_timesteps_t_r_v1(
    P_mean_t,
    P_std_t,
    P_mean_r,
    P_std_r,
    ratio,
    num_samples: int,
    device: torch.device,
    dist: str,
):
    """
    Sampler (t, r): independently sample t and r, with post-processing.
    Version 1: different post-processing to ensure t >= r.
    """
    # step 1: sample two independent timesteps
    t = _timestep_sample(P_mean_t, P_std_t, num_samples, device=device, dist=dist)
    r = _timestep_sample(P_mean_r, P_std_r, num_samples, device=device, dist=dist)

    # step 2: make t and r different with a probability of args.ratio
    prob = torch.rand(num_samples, device=device)
    mask = prob < 1 - ratio
    r = torch.where(mask, t, r)

    # step 3: ensure t >= r
    r = torch.minimum(t, r)

    return t, r
