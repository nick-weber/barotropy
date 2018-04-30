# -*- coding: utf-8 -*-

"""
Module containing the ConstantPrognostic object Forcing and some helper functions
"""

import numpy as np
from sympl import (Prognostic, ConstantPrognostic, DataArray)


class Forcing(ConstantPrognostic):
    """
    Prescribes a time-invariant vorticity tendency (e.g., Rossby
    Wave Source).
    """

    @classmethod
    def from_numpy_array(cls, tendency, linearized=False, **kwargs):
        """
        Args
        ----
        tendency : ndarray
            A 2D (lat, lon) numpy array of vorticity tendency values
            (units: s^-2) to be returned by this Prognostic.
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

        tendencies = {
            vort_varname: DataArray(
                tendency.copy(),
                dims=('lat', 'lon'),
                attrs={'units': 's^-2'})
        }

        if 'name' not in kwargs.keys() or kwargs['name'] is None:
            kwargs['name'] = 'forcing'
        return cls(tendencies, **kwargs)

    @classmethod
    def gaussian_tendencies(cls, gridlat, gridlon, centerlocs=None, amplitudes=None,
                            widths=None, latlon=True, linearized=False, **kwargs):
        """
        Creates a full grid containing one or more Gaussian vorticity tendency
        features. By default, creates a single Gaussian centered at (35N, 160E)
        with a width of 15 degrees and an amplitude of 10e-10 s^-2.

        Parameters
        ----------
        gridlat : ndarray
            A 2D or 1D array of latitudes (in degrees).
        gridlon : ndarray
            A 2D or 1D array of longitudes (in degrees).
        centerlocs : list
            A list of tuples containing (lat,lon) or (y,x) coordinates
            for the centers of the desired Gaussian forcings.
            default = [(35., 160.]), i.e. 35N,160E
        amplitudes : list
            A list of floats prescribing the amplitudes (in s^-2) for
            the desired Gaussian vorticity forcings.
            default = [10 * 10^-10]
        widths : list
            A list of floats or ints prescribing the xy widths of the
            desired Gausian forcings.
            default = [10], i.e. ten degrees
        latlon : bool
            If True, centerlocs and widths are in units of "degrees lat/lon".
            Otherwise, the units are "# of grid points".
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
            centerlocs = [(35., 160.)]
            amplitudes = [10e-10]
            widths = [15]
            latlon = True

        # TODO: Handle case where the Gaussian "block" runs into the domain edges.
        for cen, amp, wd in zip(centerlocs, amplitudes, widths):
            # Get index values for loc and wd if they were given
            # as lat/lon values
            if latlon:
                cj, ci = np.unravel_index(np.argmin(abs(gridlat-cen[0]) +
                                                    abs(gridlon-cen[1])), gridlat.shape)
                nxny = int(wd / (gridlon[0, 1] - gridlon[0, 0]))
            else:
                cj, ci = cen
                nxny = wd
            if nxny % 2 == 0:
                nxny = nxny + 1
            hw = int((nxny - 1) / 2)  # half the width

            # Create the gaussian in a nxny-by-nxny box
            x, y = np.meshgrid(np.linspace(-1, 1, nxny), np.linspace(-1, 1, nxny))
            d = np.sqrt(x**2 + y**2)
            gaus = amp * np.exp(-(d**2) / (2.*0.25**2))  # mu = 0, sigma = 0.25

            # Insert (rather, add) that gaussian into the forcing field
            forcing[cj-hw:cj+hw+1, ci-hw:ci+hw+1] += gaus

        return cls.from_numpy_array(forcing, linearized=linearized, **kwargs)
