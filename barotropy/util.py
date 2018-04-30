# -*- coding: utf-8 -*-

"""
Utility functions for use in the other modules.
"""
import numpy as np
import spharm


def gaussian_latlon_grid(nlat, as_2d=True):
    """
    Creates a gaussian lat/lon grid for spherical transforms.

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


def gaussian_blob_2d(latgrid, longrid, center, width, amplitude):
    """
    Creates a 2D field of zeros and containing a single gaussian 'blob.'

    Args
    ----
    latgrid : numpy array
        2D array of latitudes.
    longrid :
        2D array of longitudes
    center : tuple
        Two floats indicating the center latitude and longitude, respectively,
        of the gaussian blob (same units as latgrid & longrid).
    width : float
        RMS width of the gaussian blob (same units as latgrid & longrid).
    amplitude : float
        Peak value of the gaussian blob.

    Returns
    -------
    gaus : numpy array
        2D array (same shape as latgrid/longrid) including the gaussian
        blob and zeros elsewhere.
    """
    ydist = abs(latgrid - center[0])
    xdist = abs(longrid - center[1])
    xdist[xdist > 180.] = abs(xdist[xdist > 180.] - 360.)
    distance = np.sqrt(xdist ** 2 + ydist ** 2)
    gaus = amplitude * np.exp(-(distance ** 2) / (2 * width ** 2))
    return gaus


