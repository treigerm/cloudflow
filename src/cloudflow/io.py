"""Output writer for cloudflow generations."""

import numpy as np
import xarray as xr

from .normalization import denormalize_era5, denormalize_modis


def save_samples_netcdf(
    path: str,
    image_out: np.ndarray,
    image_tar: np.ndarray | None,
    era5_data: np.ndarray,
    modis_stats: dict,
    modis_stats_l1: dict,
    era5_stats: dict,
    latitude: np.ndarray,
    longitude: np.ndarray,
    solar_zenith_angle: np.ndarray | None,
    land_mask: np.ndarray | None,
    timestamp: str | None,
    era5_variables: list[str],
    era5_levels: list[int],
    output_bands: list[str],
    modis_l2_channels: list[str] | None = None,
    extra_attrs: dict | None = None,
) -> None:
    """Write predictions, ground truth, and conditioning to a NetCDF file.

    All arrays are passed in *normalised* form (the convention used inside the
    sampling code) and are de-normalised here so downstream tools see physical
    units.

    Parameters
    ----------
    path
        Output file path.
    image_out
        Generated samples, shape ``(E, C_all, H, W)``, normalised.
    image_tar
        Optional ground truth, shape ``(1, C_l1, H, W)`` or ``(C_l1, H, W)``,
        normalised (L1 bands only).
    era5_data
        Normalised ERA5 conditioning, shape ``(C_era5, H, W)``.
    modis_stats, modis_stats_l1, era5_stats
        Stats dicts returned by :mod:`cloudflow.normalization`.
    latitude, longitude
        Coordinate arrays, shape ``(H, W)``.
    solar_zenith_angle, land_mask
        Optional auxiliary 2-D arrays for context.
    timestamp
        Observation timestamp (stored as a global attribute).
    era5_variables, era5_levels, output_bands, modis_l2_channels
        Channel metadata used to build coordinate labels.
    extra_attrs
        Additional global attributes to attach (e.g. source file paths).
    """
    image_out_denorm = np.stack(
        [denormalize_modis(image_out[i], modis_stats) for i in range(image_out.shape[0])]
    )
    era5_denorm = denormalize_era5(era5_data, era5_stats)

    num_levels = len(era5_levels)
    truth_channel_names = list(output_bands)
    pred_channel_names = list(output_bands) + list(modis_l2_channels or [])

    data_vars = {
        "prediction": (("ensemble", "pred_channel", "y", "x"), image_out_denorm),
    }
    if image_tar is not None:
        if image_tar.ndim == 4:
            image_tar = image_tar[0]
        data_vars["truth"] = (
            ("truth_channel", "y", "x"),
            denormalize_modis(image_tar, modis_stats_l1),
        )
    if solar_zenith_angle is not None:
        data_vars["solar_zenith_angle"] = (("y", "x"), solar_zenith_angle)
    if land_mask is not None:
        data_vars["land_mask"] = (("y", "x"), land_mask)

    for i, var_name in enumerate(era5_variables):
        var_data = era5_denorm[i * num_levels : (i + 1) * num_levels]
        data_vars[var_name] = (("level", "y", "x"), var_data)

    coords = {
        "latitude": (("y", "x"), latitude),
        "longitude": (("y", "x"), longitude),
        "ensemble": np.arange(image_out.shape[0]),
        "truth_channel": truth_channel_names,
        "pred_channel": pred_channel_names,
        "level": era5_levels,
    }

    attrs = {}
    if timestamp is not None:
        attrs["timestamp"] = timestamp
    if extra_attrs:
        attrs.update(extra_attrs)

    ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    ds.to_netcdf(path)
