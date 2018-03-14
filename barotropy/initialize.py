# -*- coding: utf-8 -*-

"""
Module containing functions to initialize the model state.
"""
import numpy as np
import spharm
from datetime import datetime
from sympl import DataArray, add_direction_names, get_constant
from .util import buffer_poles, unbuffer_poles


def sinusoidal_perts_on_zonal_jet(idate=None, amp=12e-5, m=4, theta0=45., theta_w=15.):
    """
    Creates ICs for an idealized case: extratropical zonal jets
    with superimposed sinusoidal NH vorticity perturbations.

    Args
    ----
    idate : datetime
        Forecast initialization date
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

    # Create a 2.5-degree regular lat/lon grid
    lo = np.arange(0., 360., 2.5)
    la = np.arange(-87.5, 88., 2.5)[::-1]
    lons, lats = np.meshgrid(lo, la)
    theta = lats * np.pi/180.
    lamb = lons * np.pi/180.

    # Mean state: zonal extratropical jets
    ubar = 25 * np.cos(theta) - 30 * np.cos(theta)**3 + \
        300 * np.sin(theta)**2 * np.cos(theta)**6
    vbar = np.zeros(np.shape(ubar))

    # Get the mean state vorticity from ubar and vbar
    s = spharm.Spharmt(lamb.shape[1], lamb.shape[0]+2, gridtype='regular',
                       rsphere=get_constant('planetary_radius', 'm'), legfunc='computed')
    vortb_spec, _ = s.getvrtdivspec(buffer_poles(ubar), buffer_poles(vbar))
    vort_bar = unbuffer_poles(s.spectogrd(vortb_spec))

    # Initial perturbation: sinusoidal vorticity perturbations
    theta0 = np.deg2rad(theta0)  # center lat --> radians
    theta_w = np.deg2rad(theta_w)  # halfwidth ---> radians
    vort_prime = 0.5 * amp * np.cos(theta) * np.exp(-((theta-theta0)/theta_w)**2) * \
        np.cos(m*lamb)

    # Generate the state
    add_direction_names(x='lon', y='lat')
    ics = {
        'time': idate,
        'lat': DataArray(
            lats,
            dims=('lat', 'lon'),
            attrs={'units': 'degrees', 'gridtype': 'regular'}),
        'lon': DataArray(
            lons,
            dims=('lat', 'lon'),
            attrs={'units': 'degrees', 'gridtype': 'regular'}),
        'base_atmosphere_relative_vorticity': DataArray(
            vort_bar,
            dims=('lat', 'lon'),
            attrs={'units': 's^-1', 'gridtype': 'regular'}),
        'perturbation_atmosphere_relative_vorticity': DataArray(
            vort_prime,
            dims=('lat', 'lon'),
            attrs={'units': 's^-1', 'gridtype': 'regular'})
        }
    return ics


def from_u_and_v_winds(lats, lons, ubar, vbar, uprime, vprime,
                       ntrunc=None, idate=None):
    """
    Creates ICs from numpy arrays describing the intial wind field.

    Args
    ----
    lats : numpy array
        1D array of global, evenly spaced latitudes (in degrees)
    lons : numpy  array
        1D array of global, evenly spaced longitudes (in degrees)
    ubar : numpy  array
        2D array (nlat, nlon) of mean state zonal winds
    vbar : numpy  array
        2D array (nlat, nlon) of mean state meridional winds
    uprime : numpy  array
        2D array (nlat, nlon) of perturbation zonal winds
    vprime : numpy  array
        2D array (nlat, nlon) of perturbation meridional winds
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

    # Make sure that latitudes are symmetric about the equator
    if not np.allclose(np.abs(lats), np.abs(lats)[::-1]):
        raise ValueError('Latitudes must be symmetric about equator.')

    # If the data is oriented S-to-N, we need to reverse it
    if lats[-1] > lats[0]:
        lats = lats[::-1]
        ubar = ubar[::-1, :]
        vbar = vbar[::-1, :]
        uprime = uprime[::-1, :]
        vprime = vprime[::-1, :]

    # Make sure the poles are excluded from the data
    if lats[0] == 90.:
        lats = lats[1:-1]
        ubar = ubar[1:-1, :]
        vbar = vbar[1:-1, :]
        uprime = uprime[1:-1, :]
        vprime = vprime[1:-1, :]
    lons, lats = np.meshgrid(lons, lats)

    # Get the mean state & perturbation vorticity from the winds
    s = spharm.Spharmt(lats.shape[1], lats.shape[0] + 2, gridtype='regular',
                       rsphere=get_constant('planetary_radius', 'm'), legfunc='computed')
    vortb_spec, _ = s.getvrtdivspec(buffer_poles(ubar), buffer_poles(vbar), ntrunc=ntrunc)
    vort_bar = unbuffer_poles(s.spectogrd(vortb_spec))
    vortp_spec, _ = s.getvrtdivspec(buffer_poles(uprime), buffer_poles(vprime), ntrunc=ntrunc)
    vort_prime = unbuffer_poles(s.spectogrd(vortp_spec))

    # Generate the state
    add_direction_names(x='lon', y='lat')
    ics = {
        'time': idate,
        'lat': DataArray(
            lats,
            dims=('lat', 'lon'),
            attrs={'units': 'degrees', 'gridtype': 'regular'}),
        'lon': DataArray(
            lons,
            dims=('lat', 'lon'),
            attrs={'units': 'degrees', 'gridtype': 'regular'}),
        'base_atmosphere_relative_vorticity': DataArray(
            vort_bar,
            dims=('lat', 'lon'),
            attrs={'units': 's^-1', 'gridtype': 'regular'}),
        'perturbation_atmosphere_relative_vorticity': DataArray(
            vort_prime,
            dims=('lat', 'lon'),
            attrs={'units': 's^-1', 'gridtype': 'regular'})
    }
    return ics
