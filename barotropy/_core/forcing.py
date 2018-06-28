# -*- coding: utf-8 -*-

"""
Module containing the ConstantPrognostic object Forcing and some helper functions
"""

import numpy as np
from sympl import (ConstantPrognostic, DataArray)
from .util import gaussian_blob_2d, interp_to_gaussian
from .initialize import TRUNC_NLATS


class Forcing(ConstantPrognostic):
    """
    Prescribes a time-invariant vorticity tendency (e.g., Rossby
    Wave Source).
    """

    @classmethod
    def from_numpy_array(cls, tendency, lats, lons, ntrunc=42, linearized=False):
        """
        Args
        ----
        tendency : ndarray
            A 2D (lat, lon) numpy array of vorticity tendency values
            (units: s^-2) to be returned by this Prognostic.
        lats : numpy array
            2D array of latitudes.
        lons : numpy array
            2D array of longitudes.
        ntrunc : int
            Triangular trunction (e.g., 42 for T42).
        linearized : bool
            True is this is a linearized model (forcing tendency will
            be applied to *perturbation* vorticity).

        Raises
        ------
        ValueError
            If tendencies is not a 2D array.
        """
        if not isinstance(tendency, np.ndarray) or len(tendency.shape) != 2:
            raise ValueError('Input tendency must be a 2D numpy array.')

        if linearized:
            vort_varname = 'perturbation_atmosphere_relative_vorticity'
        else:
            vort_varname = 'atmosphere_relative_vorticity'

        if ntrunc not in TRUNC_NLATS.keys():
            raise ValueError('Truncation T{} is not in the dictionary TRUNC_NLATS'.format(ntrunc))
        nlat = TRUNC_NLATS[ntrunc]

        tendencies = {
            vort_varname: DataArray(
                interp_to_gaussian(lats, lons, tendency, nlat=nlat),
                dims=('lat', 'lon'),
                attrs={'units': 's^-2'})
        }

        return cls(tendencies)

    @classmethod
    def gaussian_tendencies(cls, gridlat, gridlon, centerlocs=None, amplitudes=None,
                            widths=None, ntrunc=42, linearized=False):
        """
        Creates a full grid containing one or more Gaussian vorticity tendency
        features. By default, creates a single Gaussian centered at (25N, 165E)
        with a width of 15 degrees and an amplitude of 10e-10 s^-2.
        Parameters
        ----------
        gridlat : ndarray
            A 2D or 1D array of latitudes (in degrees).
        gridlon : ndarray
            A 2D or 1D array of longitudes (in degrees).
        centerlocs : list
            A list of tuples containing (lat,lon) coordinates
            for the centers of the desired Gaussian forcings.
            default = [(25., 165.]), i.e. 35N,160E
        amplitudes : list
            A list of floats prescribing the amplitudes (in s^-2) for
            the desired Gaussian vorticity forcings.
            default = [10 * 10^-10]
        widths : list
            A list of floats or ints prescribing the degree widths of the
            desired Gausian forcings.
            default = [15], i.e. ten degrees
        ntrunc : int
            Triangular trunction (e.g., 42 for T42).
        linearized : bool
            True is this is a linearized model (forcing tendency will
            be applied to *perturbation* vorticity).

        Returns
        -------
        A Forcing object containing the prescribed gaussian vorticity
        tendencies.

        Raises
        ------
        ValueError
            If gridlat and gridlon are not the same shape.
        """
        if gridlat.shape != gridlon.shape:
            raise ValueError('gridlat and gridlon must be the same shape.')

        # If our grid lats/lons are not 2D arrays, create a meshgrid
        if not len(gridlat.shape) == 2:
            gridlon, gridlat = np.meshgrid(gridlon, gridlat)

        # Create an empty forcing field
        forcing = np.zeros(gridlat.shape)

        # Fill in defaults if no gaussians were provided
        if centerlocs is None or amplitudes is None or widths is None:
            centerlocs = [(25., 165.)]
            amplitudes = [10e-10]
            widths = [15]

        for cen, amp, wd in zip(centerlocs, amplitudes, widths):
            # Create the 2D gaussian field/blob
            gaus = gaussian_blob_2d(gridlat, gridlon, cen, wd, amp)

            # Insert (rather, add) that gaussian into the forcing field
            forcing += gaus

        return cls.from_numpy_array(forcing, gridlat, gridlon, ntrunc=ntrunc, linearized=linearized)
