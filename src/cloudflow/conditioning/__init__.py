"""Conditioning inputs for cloudflow.

A :class:`Conditioning` bundles everything the model needs to score-match
samples for a given ERA5 + MODIS-grid context:

* a normalised ``y_lr`` tensor with ERA5 channels (concatenated across
  variables and pressure levels) followed by surface metadata
  (``cos(solar_zenith_angle)`` and ``land_mask``);
* an ``era52modis`` index map for the compact ERA5 attention path;
* reference grid coordinates (latitude / longitude) used at output time.

Built via :meth:`Conditioning.from_modis_hdf`, which reads a MODIS HDF
granule, fetches ERA5 from ARCO-ERA5, and colocates everything.
"""

from .core import Conditioning

__all__ = ["Conditioning"]
