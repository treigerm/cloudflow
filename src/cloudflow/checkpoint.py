"""Checkpoint loading via physicsnemo's serialisation."""

import glob
import re
from pathlib import Path

import torch
from physicsnemo import Module


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a flow-matching checkpoint.

    ``path`` may point either at a ``.mdlus`` file directly, or at a directory
    containing ``UNet.0.{index}.mdlus`` files — the latest such file is loaded.
    """
    path = Path(path)
    if path.is_dir():
        path = _find_latest_checkpoint(path)
    net = Module.from_checkpoint(str(path))
    net.eval().requires_grad_(False).to(device)
    return net


def _find_latest_checkpoint(ckpt_dir: Path) -> Path:
    pattern = re.compile(r"^UNet\.0\.(\d+)\.mdlus$")
    best_index = -1
    best_path = None
    for p in glob.glob(str(ckpt_dir / "UNet.0.*.mdlus")):
        m = pattern.match(Path(p).name)
        if m:
            i = int(m.group(1))
            if i > best_index:
                best_index = i
                best_path = Path(p)
    if best_path is None:
        raise FileNotFoundError(
            f"No checkpoint files matching 'UNet.0.*.mdlus' found in {ckpt_dir}"
        )
    return best_path
