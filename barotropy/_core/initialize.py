# -*- coding: utf-8 -*-

"""
Module containing functions to initialize the model state.
"""
import numpy as np
import spharm
from datetime import datetime
from sympl import DataArray, get_constant, add_direction_names
from .util import gaussian_latlon_grid, interp_to_gaussian

Omega = get_constant('planetary_rotation_rate', 's^-1')
Re = get_constant('planetary_radius', 'm')


def super_rotation(linearized=False, idate=None, nlat=128):
    """
    Creates ICs for an idealized case: "zonal flow corresponding to
    a super-rotation of the atmosphere with a maximum value of ~15.4 m/s
    on the equator." (Sardeshmukh and Hoskins 1988)

    Args
    ----
    linearized : bool
        True if this is a linearized model.
    idate : datetime
        Forecast initialization date
    nlat : int
        Number of latitudes for the gaussian grid.

    Returns
    -------
    ics : dict
        Model initial state (a dictionary of DataArrays)
    """
    # Set the init. date if not provided
    if idate is None:
        idate = datetime.utcnow().replace(hour=0, minute=0,
                                          second=0, microsecond=0)

    # Create a gaussian lat/lon grid
    lons, lats = gaussian_latlon_grid(nlat)
    theta = lats * np.pi/180.
    lamb = lons * np.pi/180.

    # Mean state: zonal extratropical jets
    ubar = Re * Omega * np.cos(theta) / 30.
    vbar = np.zeros(ubar.shape)

    # Get the mean state vorticity from ubar and vbar
    s = spharm.Spharmt(lamb.shape[1], lamb.shape[0], gridtype='gaussian',
                       rsphere=Re, legfunc='computed')
    vortb_spec, _ = s.getvrtdivspec(ubar, vbar)
    vort_bar = s.spectogrd(vortb_spec)
    vort_prime = np.zeros(vort_bar.shape)

    # Generate the state
    return _generate_state(idate, lats, lons, vort_bar, vort_prime, linearized=linearized)


def sinusoidal_perts_on_zonal_jet(linearized=False, idate=None, nlat=128, amp=8e-5, m=4, theta0=45., theta_w=15.):
    """
    Creates ICs for an idealized case: extratropical zonal jets
    with superimposed sinusoidal NH vorticity perturbations.

    Taken from the GFDL model documentation (default case)

    Args
    ----
    linearized : bool
        True if this is a linearized model.
    idate : datetime
        Forecast initialization date
    nlat : int
        Number of latitudes for the gaussian grid.
    amp : float
        Vorticity perturbation amplitude [s^-1].
    m : int
        Vorticity perturbation zonal wavenumber.
    theta0 : float
        Center latitude [degrees] for the vorticity perturbations.
    theta_w : float
        Halfwidth [degrees lat/lon] of the vorticity perturbations.

    Returns
    -------
    ics : dict
        Model initial state (a dictionary of DataArrays)
    """
    # Set the init. date if not provided
    if idate is None:
        idate = datetime.utcnow().replace(hour=0, minute=0,
                                          second=0, microsecond=0)

    # Create a gaussian lat/lon grid
    lons, lats = gaussian_latlon_grid(nlat)
    theta = np.deg2rad(lats)
    lamb = np.deg2rad(lons)

    # Mean state: zonal extratropical jets
    ubar = 25 * np.cos(theta) - 30 * np.cos(theta)**3 + \
        300 * np.sin(theta)**2 * np.cos(theta)**6
    vbar = np.zeros(np.shape(ubar))

    # Get the mean state vorticity from ubar and vbar
    s = spharm.Spharmt(lamb.shape[1], lamb.shape[0], gridtype='gaussian',
                       rsphere=Re, legfunc='computed')
    vortb_spec, _ = s.getvrtdivspec(ubar, vbar, ntrunc=42)
    vort_bar = s.spectogrd(vortb_spec)

    # Initial perturbation: sinusoidal vorticity perturbations
    theta0 = np.deg2rad(theta0)  # center lat --> radians
    theta_w = np.deg2rad(theta_w)  # halfwidth ---> radians
    vort_prime = 0.5 * amp * np.cos(theta) * np.exp(-((theta-theta0)/theta_w)**2) * \
        np.cos(m*lamb)
    vort_prime = s.spectogrd(s.grdtospec(vort_prime, ntrunc=42))

    # Generate the state
    return _generate_state(idate, lats, lons, vort_bar, vort_prime, linearized=linearized)


def from_u_and_v_winds(lats, lons, ubar, vbar, uprime=None, vprime=None, interp=True,
                       linearized=False, ntrunc=None, idate=None):
    """
    Creates ICs from numpy arrays describing the intial wind field.

    Args
    ----
    lats : numpy array
        1D array of global latitudes (in degrees)
    lons : numpy  array
        1D array of global longitudes (in degrees)
    ubar : numpy  array
        2D array (nlat, nlon) of mean state zonal winds
    vbar : numpy  array
        2D array (nlat, nlon) of mean state meridional winds
    uprime : numpy  array
        2D array (nlat, nlon) of perturbation zonal winds
    vprime : numpy  array
        2D array (nlat, nlon) of perturbation meridional winds
    interp : bool
        If True, fields will be interpolated to a gaussian grid.
    linearized : bool
        True if this is a linearized model.
    ntrunc : int
        Triangular truncation for spherical harmonics transformation
    idate : datetime
        Foreast initialization date

    Returns
    -------
    ics : dict
        Model initial state (a dictionary of DataArrays)
    """
    # Set the init. date if not provided
    if idate is None:
        idate = datetime.utcnow().replace(hour=0, minute=0,
                                          second=0, microsecond=0)

    # If only ubar and vbar are provided, then forecast will be nonlinear
    if uprime is None or vprime is None:
        uprime = np.zeros(ubar.shape)
        vprime = np.zeros(vbar.shape)
        linearized = False

    # Interpolate to a gaussian grid, if necessary
    gridlons, gridlats = np.meshgrid(lons, lats)
    if interp:
        lats, lons, ubar = interp_to_gaussian(gridlats, gridlons, ubar, return_latlon=True)
        vbar = interp_to_gaussian(gridlats, gridlons, vbar)
        if linearized:
            uprime = interp_to_gaussian(gridlats, gridlons, uprime)
            vprime = interp_to_gaussian(gridlats, gridlons, vprime)
        else:
            uprime = np.zeros(ubar.shape)
            vprime = np.zeros(vbar.shape)
    else:
        lons, lats = gridlons, gridlats

    # # If the data is oriented S-to-N, we need to reverse it
    # if lats[-1] > lats[0]:
    #     lats = lats[::-1]
    #     ubar = ubar[::-1, :]
    #     vbar = vbar[::-1, :]
    #     uprime = uprime[::-1, :]
    #     vprime = vprime[::-1, :]
    # lons, lats = np.meshgrid(lons, lats)

    # Get the mean state & perturbation vorticity from the winds
    s = spharm.Spharmt(lats.shape[1], lats.shape[0], gridtype='gaussian',
                       rsphere=get_constant('planetary_radius', 'm'), legfunc='computed')
    vortb_spec, _ = s.getvrtdivspec(ubar, vbar, ntrunc=ntrunc)
    vort_bar = s.spectogrd(vortb_spec)
    vortp_spec, _ = s.getvrtdivspec(uprime, vprime, ntrunc=ntrunc)
    vort_prime = s.spectogrd(vortp_spec)

    # Generate the state
    return _generate_state(idate, lats, lons, vort_bar, vort_prime, linearized=linearized)


def _generate_state(idate, lats, lons, vort_bar, vort_prime, linearized=False):
    """
    Args
    ----
    idate : datetime
        Forecast initialization date.
    lats : numpy array
        2D array of latitudes.
    lons : numpy array
        2D array of longitudes.
    vort_bar : numpy array
        2D array of basic state vorticity.
    vort_prime : numpy array
        2D array of perturbation vorticity.
    linearized : bool
        True if this is a linearized model. If false, vort_bar and
        vort_prime are added together for the full vorticity field.

    Returns
    -------
    ics : dict
        Dictionary of DataArrays containing the forecast initial state.
    """

    # TODO: add assertions/checks for data shapes
    add_direction_names(x='lat', y='lon')
    ics = {
        'time': idate,
        'latitude': DataArray(
            lats,
            dims=('lat', 'lon'),
            attrs={'units': 'degrees', 'gridtype': 'gaussian'}),
        'longitude': DataArray(
            lons,
            dims=('lat', 'lon'),
            attrs={'units': 'degrees', 'gridtype': 'gaussian'})
    }

    if linearized:
        ics['base_atmosphere_relative_vorticity'] = DataArray(
            vort_bar,
            dims=('lat', 'lon'),
            attrs={'units': 's^-1', 'gridtype': 'gaussian'})

        ics['perturbation_atmosphere_relative_vorticity'] = DataArray(
            vort_prime,
            dims=('lat', 'lon'),
            attrs={'units': 's^-1', 'gridtype': 'gaussian'})
    else:
        ics['atmosphere_relative_vorticity'] = DataArray(
            vort_bar + vort_prime,
            dims=('lat', 'lon'),
            attrs={'units': 's^-1', 'gridtype': 'gaussian'})
    return ics
