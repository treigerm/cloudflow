# Cloudflow

> [!WARNING]
> **Experimental.** This repository is under active development and has
> not stabilised; expect breaking changes between commits. Raise an issue if you
> encounter problems.

CloudFlow is a conditional flow-matching model that generates MODIS-like cloud radiances 
conditioned on ERA5 atmospheric states.

## Install

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) 
installed on your system then run:
```bash
uv sync
```

## Quickstart

```python
from cloudflow import Conditioning, load_checkpoint, load_data_info, sample

info = load_data_info("checkpoints/data_info.yaml")
model = load_checkpoint("checkpoints/flow.mdlus", device="cuda")

c = info.crop_size  # e.g. 256
cond = Conditioning.from_modis_hdf(
    hdf_path="data/MYD021KM.A2023157.0820.061.hdf",
    data_info=info,
    bounds=(500, 500 + c, 800, 800 + c),
)

samples = sample(model, cond, num_ensembles=10, num_steps=30)
# samples is a torch.Tensor of shape (10, C_out, H, W).
```

Or, via the CLI:

```bash
uv run cloudflow sample \
    --ckpt checkpoints/flow.mdlus \
    --data-info checkpoints/data_info.yaml \
    --modis-hdf data/MYD021KM.A2023157.0820.061.hdf \
    --bounds 500,756,800,1056 \
    --out samples.nc
```
