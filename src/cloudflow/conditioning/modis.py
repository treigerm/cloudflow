"""MODIS L1B HDF ingest."""

import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.vectorized
import xarray as xr
from satpy import Scene

MODIS_WAVELENGTHS = {
    "1": 0.645,
    "2": 0.8585,
    "3": 0.469,
    "4": 0.555,
    "5": 1.24,
    "6": 1.64,
    "7": 2.13,
    "8": 0.4125,
    "9": 0.443,
    "10": 0.488,
    "11": 0.531,
    "12": 0.551,
    "13lo": 0.667,
    "13hi": 0.667,
    "14lo": 0.678,
    "14hi": 0.678,
    "15": 0.748,
    "16": 0.8695,
    "17": 0.905,
    "18": 0.936,
    "19": 0.940,
    "20": 3.75,
    "21": 3.959,
    "22": 3.959,
    "23": 4.05,
    "24": 4.4655,
    "25": 4.5155,
    "26": 1.375,
    "27": 6.715,
    "28": 7.325,
    "29": 8.55,
    "30": 9.73,
    "31": 11.03,
    "32": 12.02,
    "33": 13.335,
    "34": 13.635,
    "35": 13.935,
    "36": 14.235,
}


def preprocess_fn_radiances(file):
    """Read a MODIS L1B HDF granule via satpy and return an xarray.Dataset."""
    scn = Scene(reader="modis_l1b", filenames=file)
    channels = list(MODIS_WAVELENGTHS.keys())
    scn.load(channels, generate=False, calibration="radiance")
    scn.load(["satellite_zenith_angle", "solar_zenith_angle"])

    ds = scn.to_xarray_dataset()
    attrs_dict = {x: ds[x].attrs for x in channels}

    ds = ds.assign(Rad=xr.concat([ds[x] for x in channels], dim="band"))
    ds = ds.drop_vars(list(channels))
    ds = ds.assign_coords(band=list(channels))

    time_stamp = pd.to_datetime(ds.attrs["start_time"])
    time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M")
    ds = ds.assign_coords({"time": [time_stamp]})

    first = list(attrs_dict.keys())[0]
    ds.attrs = dict(
        calibration=attrs_dict[first]["calibration"],
        standard_name=attrs_dict[first]["standard_name"],
        platform_name=attrs_dict[first]["platform_name"],
        sensor=attrs_dict[first]["sensor"],
        units=attrs_dict[first]["units"],
    )
    ds = ds.assign_coords({"band_wavelength": list(MODIS_WAVELENGTHS.values())})
    return ds


def create_land_mask_cartopy(ds, resolution: str = "110m") -> np.ndarray:
    """Build a 0/1 land mask on the MODIS grid using cartopy Natural Earth."""
    lats = ds.latitude.values
    lons = ds.longitude.values

    land_feature = cfeature.LAND.with_scale(resolution)
    land_multipolygon = shapely.geometry.MultiPolygon(
        [g for g in land_feature.geometries() if g.geom_type in ("Polygon", "MultiPolygon")]
    )
    mask = shapely.vectorized.contains(land_multipolygon, lons.flatten(), lats.flatten())
    return mask.reshape(lats.shape).astype(np.uint8)


def orient_tile(tile_ds):
    """Rotate a tile 180° so NW is at [0, 0]."""
    for var_name in tile_ds.data_vars:
        arr = tile_ds[var_name].values
        if arr.ndim == 2:
            tile_ds[var_name].values = np.rot90(arr, k=2)
        elif arr.ndim == 3:
            tile_ds[var_name].values = np.rot90(arr, k=2, axes=(1, 2))
    return tile_ds


def load_modis_data(
    hdf_path: str,
    bounds: tuple[int, int, int, int],
    output_bands: list[str],
) -> tuple[xr.Dataset, str]:
    """Load a MODIS HDF granule, crop, orient, and add a land mask.

    Returns
    -------
    ds
        ``xarray.Dataset`` with ``Rad`` (shape ``(C_l1, H, W)``),
        ``solar_zenith_angle``, ``land_mask``, and coordinate variables
        ``latitude``/``longitude``.
    timestamp
        Observation time, in ISO format.
    """
    x_min, x_max, y_min, y_max = bounds

    ds = preprocess_fn_radiances([hdf_path])
    ds = ds.sel(band=list(output_bands))
    ds = ds.isel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    ds = ds.reset_coords(["latitude", "longitude"])
    ds = ds.compute()

    land_mask = create_land_mask_cartopy(ds, resolution="110m")
    ds["land_mask"] = (("y", "x"), land_mask)

    ds = orient_tile(ds)
    timestamp = str(ds.coords["time"].values[0])
    return ds, timestamp
