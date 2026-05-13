"""Per-checkpoint metadata loaded from a ``data_info.yaml`` sidecar.

Each trained cloudflow checkpoint ships with a YAML file describing the
training-time data setup — which ERA5 variables and pressure levels were
used, which MODIS bands the model predicts, what auxiliary surface channels
were present, and the training patch size. cloudflow reads everything it
needs at inference from this file, so users only have to point at the
checkpoint and its ``data_info.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


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
