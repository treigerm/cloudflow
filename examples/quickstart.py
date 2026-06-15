"""Generate samples from a MODIS HDF granule and ERA5.

Reproduces the paper inference pipeline.

Usage
-----
    uv run python examples/quickstart.py \\
        --ckpt checkpoints/cloudflow \\
        --hdf data/MYD021KM.A2023157.0820.061.hdf \\
        --out samples.nc
"""

import argparse

from cloudflow import Conditioning, load_model, sample
from cloudflow.io import save_samples_netcdf
from cloudflow.normalization import load_era5_stats, load_modis_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    p.add_argument("--hdf", required=True)
    p.add_argument("--out", default="samples.nc")
    p.add_argument("--x-min", type=int, default=500)
    p.add_argument("--y-min", type=int, default=800)
    p.add_argument("--ensembles", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    model, info = load_model(args.ckpt, device=args.device)
    bounds = (
        args.x_min,
        args.x_min + info.crop_size,
        args.y_min,
        args.y_min + info.crop_size,
    )

    cond = Conditioning.from_modis_hdf(
        hdf_path=args.hdf,
        data_info=info,
        bounds=bounds,
    )

    samples = sample(
        model,
        cond,
        num_ensembles=args.ensembles,
        num_steps=args.steps,
    )

    modis_stats_l1 = load_modis_stats(info.output_bands)
    modis_stats_full = load_modis_stats(info.output_bands, info.modis_l2_channels or None)
    era5_stats_d = load_era5_stats(info.era5_variables, info.era5_levels)

    n_era5 = len(info.era5_variables) * len(info.era5_levels)
    era5_norm = cond.y_lr[0, :n_era5].cpu().numpy()

    save_samples_netcdf(
        path=args.out,
        image_out=samples.cpu().numpy(),
        image_tar=cond.truth.cpu().numpy() if cond.truth is not None else None,
        era5_data=era5_norm,
        modis_stats=modis_stats_full,
        modis_stats_l1=modis_stats_l1,
        era5_stats=era5_stats_d,
        latitude=cond.latitude,
        longitude=cond.longitude,
        solar_zenith_angle=cond.solar_zenith_angle,
        land_mask=cond.land_mask,
        timestamp=cond.timestamp,
        era5_variables=info.era5_variables,
        era5_levels=info.era5_levels,
        output_bands=info.output_bands,
        modis_l2_channels=info.modis_l2_channels or None,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
