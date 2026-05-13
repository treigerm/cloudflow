"""ERA5 fetch + colocation onto a MODIS-style grid.

Ported from corrdiff/generate_flow_patched.py:_to_radians, _get_era5_tree,
_closest_era5_time, load_era5_data.
"""

import numpy as np


def _to_radians(deg):
    return deg * np.pi / 180


def _get_era5_tree(latitudes: np.ndarray, longitudes: np.ndarray):
    import scipy.spatial

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    lat_rad = _to_radians(lat_grid.flatten())
    lon_rad = _to_radians(lon_grid.flatten())
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    coords = np.column_stack([x, y, z])
    return scipy.spatial.cKDTree(coords)


def _closest_era5_time(timestamp, era5_times):
    import pandas as pd

    timestamp_dt = pd.to_datetime(timestamp)
    timestamp_hourly = timestamp_dt.floor("h")
    era5_times = pd.to_datetime(era5_times)
    matching = era5_times[era5_times == timestamp_hourly]
    if len(matching) == 0:
        raise ValueError(f"No ERA5 data available for timestamp {timestamp_hourly}")
    return matching[0]


def load_era5_data(
    timestamp: str,
    modis_lat: np.ndarray,
    modis_lon: np.ndarray,
    variables: list[str],
    levels: list[int],
    zarr_url: str = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    zarr_storage_options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch ERA5 from a zarr store and colocate it with the MODIS grid.

    Returns
    -------
    extracted
        Array of shape ``(C, H, W)`` with C = len(variables) * len(levels).
    era52modis
        1-D array of length ``H*W`` mapping each MODIS pixel to its
        position in the unique ERA5 cell list (i.e. the inverse index used
        for compact attention).
    """
    import xarray as xr

    if zarr_storage_options is None:
        zarr_storage_options = {"token": "anon"}

    era5 = xr.open_zarr(zarr_url, chunks=None, storage_options=zarr_storage_options)
    era5 = era5[list(variables)]

    tree = _get_era5_tree(era5.latitude.values, era5.longitude.values)

    lat_rad = _to_radians(modis_lat.flatten())
    lon_rad = _to_radians(modis_lon.flatten())
    modis_coords = np.column_stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )

    _, indices = tree.query(modis_coords)

    unique_indices = np.unique(indices)
    old2new = {old: new for new, old in enumerate(unique_indices)}
    new_indices = np.array([old2new[i] for i in indices])

    era5_shape = (len(era5.latitude), len(era5.longitude))
    unique_i = unique_indices // era5_shape[1]
    unique_j = unique_indices % era5_shape[1]

    closest_time = _closest_era5_time(timestamp, era5["time"].values)
    era5_subset = era5.sel(time=closest_time).sel(level=list(levels))
    era5_subset = era5_subset.isel(
        latitude=xr.DataArray(unique_i),
        longitude=xr.DataArray(unique_j),
    )

    extracted = (
        xr.concat([era5_subset[v] for v in variables], dim="level").compute().values
    )  # (C, n_unique)

    extracted = extracted[:, new_indices]  # (C, H*W)
    extracted = extracted.reshape((-1,) + modis_lat.shape)  # (C, H, W)

    return extracted, new_indices
