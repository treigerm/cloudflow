"""Checkpoint loading via physicsnemo's serialisation.

Also defines :class:`DataInfo`, the per-checkpoint metadata loaded from a
``data_info.yaml`` sidecar. Each trained cloudflow checkpoint ships with such
a file describing the training-time data setup — which ERA5 variables and
pressure levels were used, which MODIS bands the model predicts, what
auxiliary surface channels were present, and the training patch size.
cloudflow reads everything it needs at inference from this file, so users only
have to point at the checkpoint directory.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from physicsnemo import Module


@dataclass(frozen=True)
class DataInfo:
    """Per-checkpoint dataset/preprocessing metadata."""

    era5_variables: list[str]
    era5_levels: list[int]
    output_bands: list[str]
    crop_size: int
    modis_l2_channels: list[str] = field(default_factory=list)
    omit_solar_zenith_angle: bool = False
    destriped: bool = False


def load_data_info(path: str | Path) -> DataInfo:
    """Parse a ``data_info.yaml`` file into a :class:`DataInfo`.

    Required keys: ``era5_variables``, ``era5_levels``, ``output_bands``,
    ``crop_size``. Optional keys: ``modis_l2_channels``,
    ``omit_solar_zenith_angle``, ``destriped``.
    """
    with open(path) as f:
        d = yaml.safe_load(f)

    missing = [
        k for k in ("era5_variables", "era5_levels", "output_bands", "crop_size") if k not in d
    ]
    if missing:
        raise ValueError(f"data_info.yaml at {path} is missing keys: {missing}")

    return DataInfo(
        era5_variables=[str(v) for v in d["era5_variables"]],
        era5_levels=[int(v) for v in d["era5_levels"]],
        output_bands=[str(v) for v in d["output_bands"]],
        crop_size=int(d["crop_size"]),
        modis_l2_channels=[str(v) for v in (d.get("modis_l2_channels") or [])],
        omit_solar_zenith_angle=bool(d.get("omit_solar_zenith_angle", False)),
        destriped=bool(d.get("destriped", False)),
    )


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[Module, DataInfo]:
    """Load a cloudflow checkpoint and its metadata from a checkpoint directory.

    ``path`` is a checkpoint directory containing both the model weights
    (``UNet.0.{index}.mdlus`` files — the latest is used) and a
    ``data_info.yaml`` sidecar describing the training-time data setup. A
    ``.mdlus`` file may also be passed directly, in which case the
    ``data_info.yaml`` next to it is used.

    Returns ``(model, info)``::

        model, info = load_model("checkpoints/cloudflow")
    """
    path = Path(path)
    ckpt_dir = path if path.is_dir() else path.parent
    info = load_data_info(ckpt_dir / "data_info.yaml")
    model = load_checkpoint(path, device=device)
    return model, info


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a flow-matching checkpoint.

    ``path`` may point either at a ``.mdlus`` file directly, or at a directory
    containing ``UNet.0.{index}.mdlus`` files — the latest such file is loaded.
    """
    path = Path(path)
    if path.is_dir():
        path = _find_latest_checkpoint(path)
    net = Module.from_checkpoint(str(path))
    net.eval().requires_grad_(False).to(device)
    return net


def _find_latest_checkpoint(ckpt_dir: Path) -> Path:
    pattern = re.compile(r"^UNet\.0\.(\d+)\.mdlus$")
    best_index = -1
    best_path = None
    for p in glob.glob(str(ckpt_dir / "UNet.0.*.mdlus")):
        m = pattern.match(Path(p).name)
        if m:
            i = int(m.group(1))
            if i > best_index:
                best_index = i
                best_path = Path(p)
    if best_path is None:
        raise FileNotFoundError(
            f"No checkpoint files matching 'UNet.0.*.mdlus' found in {ckpt_dir}"
        )
    return best_path
