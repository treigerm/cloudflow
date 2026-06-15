"""Compact attention indices for de-duplicated ERA5 cross-attention."""

import torch


def compute_compact_indices(gather_ixs: torch.Tensor):
    """Build representative and inverse indices for compact ERA5 attention.

    For each batch element, finds unique ERA5 cells and builds:
      - ``representative_pixels``: one pixel index per unique cell, padded to
        ``max_unique`` across the batch
      - ``inverse_ixs``: maps each pixel to its position in the compact
        representation

    Runs on whatever device the input is on. Must be called outside
    ``torch.compile`` because ``torch.unique`` is not compile-compatible.

    Parameters
    ----------
    gather_ixs : torch.Tensor
        Long tensor of shape ``(B, H*W)`` mapping each pixel to an ERA5 cell
        index.

    Returns
    -------
    representative_pixels : torch.Tensor
        Long tensor of shape ``(B, max_unique)``, zero-padded.
    inverse_ixs : torch.Tensor
        Long tensor of shape ``(B, H*W)``.
    """
    B, HW = gather_ixs.shape
    device = gather_ixs.device
    all_rep = []
    all_inv = []
    for b in range(B):
        unique_vals, inverse = torch.unique(gather_ixs[b], return_inverse=True)
        n_unique = unique_vals.shape[0]
        rep = torch.zeros(n_unique, dtype=torch.long, device=device)
        rep.scatter_(0, inverse, torch.arange(HW, device=device))
        all_rep.append(rep)
        all_inv.append(inverse)

    max_unique = max(r.shape[0] for r in all_rep)
    representative_pixels = torch.zeros(B, max_unique, dtype=torch.long, device=device)
    for b in range(B):
        n = all_rep[b].shape[0]
        representative_pixels[b, :n] = all_rep[b]
    inverse_ixs = torch.stack(all_inv)

    return representative_pixels, inverse_ixs
