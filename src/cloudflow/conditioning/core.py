"""Conditioning dataclass and constructors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data_info import DataInfo, load_data_info


@dataclass
class Conditioning:
    """A single ERA5+metadata conditioning context bound to a target grid.

    Attributes
    ----------
    y_lr
        Normalised conditioning tensor, shape ``(1, C_in, H, W)`` on the
        target device. Channel order is ``[ERA5 vars × levels (var-major or
        level-major depending on model.version2), surface metadata...]``.
    era52modis
        Long tensor of shape ``(H*W,)`` mapping each MODIS pixel to its
        nearest unique ERA5 cell. Used by compact attention. ``None`` when
        the model was not trained with compact attention.
    latitude, longitude
        Target-grid coordinates as ``(H, W)`` numpy arrays. Optional but
        recommended; written to the output NetCDF.
    image_shape
        ``(H, W)`` — convenience accessor that always matches ``y_lr``.
    timestamp
        Optional observation timestamp string.
    era5_variables
        Names of the ERA5 variables present in ``y_lr``, in order.
    era5_levels
        Pressure levels (hPa) present in ``y_lr``, in order.
    metadata_channels
        Names of the surface metadata channels appended after ERA5, in order.
    truth
        Optional ground-truth MODIS L1 tensor (normalised), shape
        ``(1, C_l1, H, W)`` — when available, written to the output for
        comparison.
    solar_zenith_angle, land_mask
        Optional auxiliary ``(H, W)`` numpy arrays written to the output.
    """

    y_lr: torch.Tensor
    era52modis: torch.Tensor | None = None
    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None
    timestamp: str | None = None
    era5_variables: list[str] = field(default_factory=list)
    era5_levels: list[int] = field(default_factory=list)
    metadata_channels: list[str] = field(default_factory=list)
    truth: torch.Tensor | None = None
    solar_zenith_angle: np.ndarray | None = None
    land_mask: np.ndarray | None = None

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(self.y_lr.shape[-2:])

    def check_compatible(self, model: torch.nn.Module) -> None:
        """Validate that this conditioning matches the model's expectations.

        Raises ``ValueError`` if the conditioning grid size differs from the
        model's training patch size (cloudflow does not patch internally), or
        if the channel count of ``y_lr`` does not match ``model.img_in_channels``.
        """
        expected_hw = (int(model.img_shape_y), int(model.img_shape_x))
        if self.image_shape != expected_hw:
            raise ValueError(
                f"Conditioning grid {self.image_shape} does not match model "
                f"patch size {expected_hw}. cloudflow generates one patch at "
                f"a time — please crop the conditioning to the model size."
            )
        if hasattr(model, "img_in_channels") and self.y_lr.shape[1] != model.img_in_channels:
            raise ValueError(
                f"Conditioning has {self.y_lr.shape[1]} channels but model "
                f"expects {model.img_in_channels}."
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_modis_hdf(
        cls,
        hdf_path: str,
        data_info: DataInfo | str | Path,
        bounds: tuple[int, int, int, int],
        zarr_url: str = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        zarr_storage_options: dict | None = None,
    ) -> Conditioning:
        """Build a Conditioning by ingesting MODIS + fetching ERA5 on the fly.

        Reproduces the paper inference pipeline: reads a MODIS L1B HDF
        granule, crops to ``bounds = (x_min, x_max, y_min, y_max)``, builds a
        land mask, fetches ERA5 from the ARCO-ERA5 zarr at the nearest hour,
        and colocates each MODIS pixel to its nearest ERA5 cell.

        ``data_info`` may be a :class:`DataInfo` or a path to a
        ``data_info.yaml`` file (which is loaded eagerly). Everything that
        depends on the trained model — ERA5 variables and levels, MODIS
        output bands, optional L2 channels, and whether the surface metadata
        includes ``cos(solar_zenith_angle)`` — comes from there.
        """
        from ..normalization import (
            load_era5_stats,
            load_modis_stats,
            normalize_era5,
            normalize_modis,
        )
        from .era5 import load_era5_data
        from .modis import load_modis_data

        if not isinstance(data_info, DataInfo):
            data_info = load_data_info(data_info)

        x_min, x_max, y_min, y_max = bounds
        crop_shape = (y_max - y_min, x_max - x_min)
        expected = (data_info.crop_size, data_info.crop_size)
        if crop_shape != expected:
            raise ValueError(
                f"bounds produce a crop of shape {crop_shape} (y, x) but the "
                f"model expects {expected} (square crop_size={data_info.crop_size}). "
                f"Adjust bounds so that both side lengths equal crop_size."
            )

        modis_ds, timestamp = load_modis_data(hdf_path, bounds, data_info.output_bands)
        era5_data, era52modis = load_era5_data(
            timestamp,
            modis_ds.latitude.values,
            modis_ds.longitude.values,
            data_info.era5_variables,
            data_info.era5_levels,
            zarr_url=zarr_url,
            zarr_storage_options=zarr_storage_options,
        )

        modis_stats_l1 = load_modis_stats(data_info.output_bands)
        era5_stats = load_era5_stats(data_info.era5_variables, data_info.era5_levels)

        modis_norm = normalize_modis(modis_ds.Rad.values, modis_stats_l1)
        era5_norm = normalize_era5(era5_data, era5_stats)

        metadata_parts: list[np.ndarray] = []
        metadata_channel_names: list[str] = []
        if not data_info.omit_solar_zenith_angle:
            metadata_parts.append(np.cos(modis_ds.solar_zenith_angle.values).astype(np.float32))
            metadata_channel_names.append("cos_solar_zenith_angle")
        metadata_parts.append(modis_ds.land_mask.values.astype(np.float32))
        metadata_channel_names.append("land_mask")
        metadata = np.stack(metadata_parts, axis=0)

        y_lr_np = np.concatenate([era5_norm, metadata], axis=0).astype(np.float32)
        y_lr = torch.from_numpy(y_lr_np).unsqueeze(0)
        truth = torch.from_numpy(modis_norm.astype(np.float32)).unsqueeze(0)

        return cls(
            y_lr=y_lr,
            era52modis=torch.from_numpy(era52modis).long(),
            latitude=modis_ds.latitude.values,
            longitude=modis_ds.longitude.values,
            timestamp=timestamp,
            era5_variables=list(data_info.era5_variables),
            era5_levels=list(data_info.era5_levels),
            metadata_channels=metadata_channel_names,
            truth=truth,
            solar_zenith_angle=modis_ds.solar_zenith_angle.values,
            land_mask=modis_ds.land_mask.values,
        )

    def to(self, device) -> Conditioning:
        """Return a copy with tensors moved to ``device``."""
        return Conditioning(
            y_lr=self.y_lr.to(device),
            era52modis=None if self.era52modis is None else self.era52modis.to(device),
            latitude=self.latitude,
            longitude=self.longitude,
            timestamp=self.timestamp,
            era5_variables=self.era5_variables,
            era5_levels=self.era5_levels,
            metadata_channels=self.metadata_channels,
            truth=None if self.truth is None else self.truth.to(device),
            solar_zenith_angle=self.solar_zenith_angle,
            land_mask=self.land_mask,
        )
