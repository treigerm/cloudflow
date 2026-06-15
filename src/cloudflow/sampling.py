"""Euler ODE sampler for flow matching (single-patch, non-tiled).

Each call generates a single image at the model's native patch size. Patched
generation is intentionally not supported; users who need larger regions can
call :func:`cloudflow.sample` repeatedly with their own tiling strategy.
"""

from collections.abc import Callable

import torch


def v_pred_from_x_pred(net: torch.nn.Module) -> Callable:
    """Wrap a network whose output is ``x_pred`` so it returns a velocity."""

    def v_pred(y_t, y_lr, t, **net_kwargs):
        y_pred = net(y_t, y_lr, t, **net_kwargs)
        return (y_pred - y_t) / (1 - t)

    return v_pred


def euler_sampler(v_pred, y_lr, shape, num_steps, device, **net_kwargs):
    """Euler ODE integration from t=0 (noise) to t=1 (data)."""
    dt = 1.0 / num_steps
    y_t = torch.randn(shape, device=device)
    for i in range(num_steps):
        t = torch.full((shape[0], 1, 1, 1), i * dt, device=device)
        v_t = v_pred(y_t, y_lr, t, **net_kwargs)
        y_t = y_t + dt * v_t
    return y_t
