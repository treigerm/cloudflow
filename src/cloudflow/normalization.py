"""MODIS and ERA5 normalization helpers.

The training pipeline normalises MODIS radiances to ``[0, 1]`` using per-band
min/max bounds, and standardises ERA5 variables using per-variable mean/std.
The corresponding stats files are bundled inside this package
(``cloudflow/stats/{modis_band_stats,era5_stats}.json``) and used
automatically — callers do not need to pass them in.
"""

import json
from collections import namedtuple
from importlib import resources

import numpy as np

Bound = namedtuple("Bound", "min max")

# MODIS L2 cloud retrievals — fixed physical bounds (kept in sync with corrdiff).
MODIS_L2_BOUNDS = {
    "cloud_top_temperature": Bound(150, 350),
    "cloud_top_height": Bound(1, 18_000),
    "cloud_water_path": Bound(0, 9_000),
    "cloud_optical_thickness": Bound(0, 100),
}


def _load_bundled_json(name: str) -> dict:
    with resources.files("cloudflow.stats").joinpath(name).open("r") as f:
        return json.load(f)


def load_modis_stats(
    output_bands: list[str],
    modis_l2_channels: list[str] | None = None,
    stats_path: str | None = None,
) -> dict:
    """Load MODIS per-band min/max bounds.

    By default reads the stats bundled with the package
    (``cloudflow/stats/modis_band_stats.json``). Pass ``stats_path`` to
    override with a custom file. L2 channels are appended using the fixed
    physical bounds in :data:`MODIS_L2_BOUNDS`.
    """
    if stats_path is None:
        stats = _load_bundled_json("modis_band_stats.json")
    else:
        with open(stats_path) as f:
            stats = json.load(f)
    band_ixs = [stats["bands"].index(b) for b in output_bands]
    mins = [stats["min"][i] for i in band_ixs]
    maxs = [stats["max"][i] for i in band_ixs]

    if modis_l2_channels:
        for ch in modis_l2_channels:
            bounds = MODIS_L2_BOUNDS[ch]
            mins.append(bounds.min)
            maxs.append(bounds.max)

    return {"min": np.array(mins), "max": np.array(maxs)}


def normalize_modis(x: np.ndarray, stats: dict) -> np.ndarray:
    mins = stats["min"][:, None, None]
    maxs = stats["max"][:, None, None]
    return (x - mins) / (maxs - mins)


def denormalize_modis(x: np.ndarray, stats: dict) -> np.ndarray:
    mins = stats["min"][:, None, None]
    maxs = stats["max"][:, None, None]
    return x * (maxs - mins) + mins


def load_era5_stats(
    variables: list[str],
    levels: list[int],
    stats_path: str | None = None,
) -> dict:
    """Load ERA5 per-variable mean/std and broadcast across pressure levels.

    By default reads the stats bundled with the package
    (``cloudflow/stats/era5_stats.json``). Pass ``stats_path`` to override
    with a custom file.
    """
    if stats_path is None:
        stats = _load_bundled_json("era5_stats.json")
    else:
        with open(stats_path) as f:
            stats = json.load(f)
    means = np.concatenate([np.full(len(levels), stats[v]["mean"]) for v in variables])
    stds = np.concatenate([np.full(len(levels), stats[v]["std"]) for v in variables])
    return {"mean": means, "std": stds}


def normalize_era5(x: np.ndarray, stats: dict) -> np.ndarray:
    means = stats["mean"][:, None, None]
    stds = stats["std"][:, None, None]
    return (x - means) / stds


def denormalize_era5(x: np.ndarray, stats: dict) -> np.ndarray:
    means = stats["mean"][:, None, None]
    stds = stats["std"][:, None, None]
    return x * stds + means
