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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cloudflow import Conditioning, load_model, sample
from cloudflow.io import save_samples_netcdf
from cloudflow.normalization import load_era5_stats, load_modis_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    p.add_argument("--hdf", required=True)
    p.add_argument("--out", default="samples.nc")
    p.add_argument("--plot", default=None, help="PNG path for the RGB comparison plot.")
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

    samples_np = samples.cpu().numpy()
    truth_np = cond.truth.cpu().numpy() if cond.truth is not None else None

    save_samples_netcdf(
        path=args.out,
        image_out=samples_np,
        image_tar=truth_np,
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

    plot_path = args.plot or str(Path(args.out).with_suffix(".png"))
    plot_rgb(samples_np, truth_np, info.output_bands, plot_path)


def plot_rgb(samples, truth, output_bands, out_path, n_cols=5):
    """Plot the truth true-color RGB next to every generated ensemble member.

    MODIS true-color uses band 1 (red, 0.645 µm), band 4 (green, 0.555 µm) and
    band 3 (blue, 0.469 µm). The model's output is already in roughly [0, 1]
    since :func:`cloudflow.normalization.normalize_modis` rescales each band by
    its min/max bounds, so we just clip and ``imshow``.
    """
    rgb_bands = ("1", "4", "3")
    if not all(b in output_bands for b in rgb_bands):
        print(f"Skipping RGB plot: output_bands {output_bands} missing one of {rgb_bands}.")
        return
    rgb_idx = [output_bands.index(b) for b in rgb_bands]

    def to_rgb(image):
        # (C, H, W) -> (H, W, 3), clipped to [0, 1] for display
        return np.clip(image[rgb_idx], 0.0, 1.0).transpose(1, 2, 0)

    panels: list[tuple[str, np.ndarray]] = []
    if truth is not None:
        panels.append(("Truth", to_rgb(truth[0])))
    panels.extend((f"Generated (ens={i})", to_rgb(samples[i])) for i in range(samples.shape[0]))

    n_cols = min(n_cols, len(panels))
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    flat_axes = axes.ravel()
    for ax, (title, img) in zip(flat_axes, panels, strict=False):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for ax in flat_axes[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
