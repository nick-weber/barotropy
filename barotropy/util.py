# -*- coding: utf-8 -*-

"""
Utility functions for use in the other modules.
"""
import numpy as np
import spharm


def gaussian_grid(nlat, as_2d=True):
    """
    Args
    ----
    nlat : int
        Number of gaussian latitudes. (# lons = 2*nlat)
    as_2d : bool
        If True, returns 2d lat/lon grids. Otherwise, returns
        1d arrays.

    Returns
    -------
    lo, la : numpy array
        Arrays of lon/lat values in degrees.
    """
    lo = np.linspace(0., 360., 2 * nlat + 1)[:-1]
    la, _ = spharm.gaussian_lats_wts(nlat)
    if as_2d:
        lo, la = np.meshgrid(lo, la)
    return lo, la


