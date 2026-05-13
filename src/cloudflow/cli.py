"""Thin Typer CLI for cloudflow.

The CLI is a single ``sample`` command that wraps the Python API for the
MODIS-HDF reproduction case. For anything more elaborate (custom inputs,
batched sweeps, plotting), import :mod:`cloudflow` and write a script.
"""

from pathlib import Path

import typer

from . import Conditioning, load_checkpoint, load_data_info, sample
from .io import save_samples_netcdf
from .normalization import load_era5_stats, load_modis_stats

app = typer.Typer(add_completion=False, help=__doc__)


def _parse_bounds(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("bounds must be 'x_min,x_max,y_min,y_max'")
    return tuple(parts)


@app.command("sample")
def sample_cmd(
    ckpt: Path = typer.Option(..., help="Path to a .mdlus checkpoint or a checkpoint directory."),
    data_info: Path = typer.Option(..., help="Path to the checkpoint's data_info.yaml."),
    modis_hdf: Path = typer.Option(..., help="MODIS L1B HDF granule."),
    bounds: str = typer.Option(..., help="Spatial crop as 'x_min,x_max,y_min,y_max'."),
    out: Path = typer.Option(..., help="Output NetCDF path."),
    ensembles: int = typer.Option(10, help="Number of ensemble members."),
    steps: int = typer.Option(30, help="ODE integration steps."),
    seed: int | None = typer.Option(None, help="RNG seed for initial noise."),
    device: str = typer.Option("cuda", help="Device: 'cuda' or 'cpu'."),
):
    """Generate flow-matching samples from a MODIS HDF + ERA5 conditioning."""
    bounds_t = _parse_bounds(bounds)

    typer.echo(f"Loading data_info {data_info}")
    info = load_data_info(data_info)

    typer.echo(f"Loading checkpoint {ckpt}")
    model = load_checkpoint(ckpt, device=device)

    typer.echo(f"Building conditioning from {modis_hdf}")
    cond = Conditioning.from_modis_hdf(
        hdf_path=str(modis_hdf),
        data_info=info,
        bounds=bounds_t,
    )

    typer.echo(f"Sampling {ensembles}×{steps}")
    samples = sample(
        model,
        cond,
        num_ensembles=ensembles,
        num_steps=steps,
        seed=seed,
    )

    typer.echo(f"Writing {out}")
    modis_stats_l1 = load_modis_stats(info.output_bands)
    modis_stats_full = load_modis_stats(info.output_bands, info.modis_l2_channels or None)
    era5_stats_d = load_era5_stats(info.era5_variables, info.era5_levels)

    # y_lr packs era5 channels first; pull only that slice for the writer.
    n_era5 = len(info.era5_variables) * len(info.era5_levels)
    era5_norm = cond.y_lr[0, :n_era5].cpu().numpy()

    save_samples_netcdf(
        path=str(out),
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
        extra_attrs={"hdf_path": str(modis_hdf), "checkpoint": str(ckpt)},
    )
    typer.echo("Done.")


if __name__ == "__main__":
    app()
