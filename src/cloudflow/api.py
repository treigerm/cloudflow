"""Top-level :func:`sample` entry point."""

import torch

from .compact import compute_compact_indices
from .conditioning import Conditioning
from .sampling import euler_sampler, v_pred_from_x_pred


def sample(
    model: torch.nn.Module,
    conditioning: Conditioning,
    num_ensembles: int = 1,
    num_steps: int = 30,
    device: torch.device | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate ensemble samples from a trained flow-matching model.

    Uses the Euler sampler over a network that predicts ``x_pred``. Other
    sampler / prediction conventions are intentionally not exposed in the
    public API.

    Parameters
    ----------
    model
        Module returned by :func:`cloudflow.load_model`.
    conditioning
        Built via :class:`Conditioning.from_modis_hdf`,
        :class:`Conditioning.from_xarray`, or :class:`Conditioning.from_tensor`.
    num_ensembles
        Number of independent samples to generate.
    num_steps
        ODE integration steps.
    device
        Target device. If ``None``, the model's device is used.
    seed
        Optional RNG seed for the initial noise.

    Returns
    -------
    torch.Tensor
        Samples of shape ``(num_ensembles, C_out, H, W)`` on ``device``.
    """
    if device is None:
        device = next(model.parameters()).device
    conditioning = conditioning.to(device)
    conditioning.check_compatible(model)

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    y_lr = conditioning.y_lr.repeat_interleave(num_ensembles, dim=0)
    H, W = conditioning.image_shape
    latent_shape = (num_ensembles, int(model.img_out_channels), H, W)

    net_kwargs = {}
    if conditioning.era52modis is not None:
        era52modis = conditioning.era52modis.reshape(1, -1).repeat(num_ensembles, 1)
        compact_rep, compact_inv = compute_compact_indices(era52modis)
        net_kwargs["compact_representative"] = compact_rep
        net_kwargs["compact_inverse"] = compact_inv

    v_pred_fn = v_pred_from_x_pred(model)

    with torch.no_grad():
        return euler_sampler(v_pred_fn, y_lr, latent_shape, num_steps, device, **net_kwargs)
