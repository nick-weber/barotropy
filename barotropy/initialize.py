# -*- coding: utf-8 -*-

"""
Module containing functions to initialize the model state.
"""
import numpy as np
import spharm
from datetime import datetime
from sympl import DataArray, add_direction_names
import namelist as NL

def sinusoidal_NH(idate=None, A=12e-5, m=4, theta0=45., thetaW=15.):
    """
    Creates ICs for an idealized case: extratropical zonal jets 
    with superimposed sinusoidal NH vorticity perturbations.
    
    Args
    ----
    idate : datetime
        Forecast initialization date
    A : float
        Vorticity perturbation amplitude [s^-1].
    m : int
        Vorticity perturbation zonal wavenumber.
    theta0 : float
        Center latitude [degrees] for the vorticity perturbations.
    thetaW : float
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
    lo = np.arange(0, 359, 2.5)
    la = np.arange(-90., 90.1, 2.5)[::-1]
    lons, lats = np.meshgrid(lo, la)
    theta = lats * np.pi/180.
    lamb = lons * np.pi/180.
    
    # Mean state: zonal extratropical jets
    ubar = 25 * np.cos(theta) - 30 * np.cos(theta)**3 + \
           300 * np.sin(theta)**2 * np.cos(theta)**6
    vbar = np.zeros(np.shape(ubar))

    # Get the mean state vorticity from Ubar and Vbar
    # TODO: Get rsphere from a namelist
    s = spharm.Spharmt(lamb.shape[1], lamb.shape[0], rsphere=6378100.,
                       gridtype='regular', legfunc='computed')
    vortb_spec, _ = s.getvrtdivspec(ubar, vbar)
    vort_bar = s.spectogrd(vortb_spec)

    # Initial perturbation: sinusoidal vorticity perturbations
    theta0 = np.deg2rad(theta0)  # center lat --> radians
    thetaW = np.deg2rad(thetaW)  # halfwidth ---> radians
    vort_prime = 0.5 * A * np.cos(theta) * np.exp(-((theta-theta0)/thetaW)**2) * \
                np.cos(m*lamb)
    
    # Generate the state
    add_direction_names(x='lon', y='lat')
    ics = {
        'time' : idate,
        'grid_latitude': DataArray(
            lats, 
            dims=('lat','lon'),
            attrs={'units': 'degrees', 'gridtype': 'regular'}),
        'grid_longitude' : DataArray(
            lons, 
            dims=('lat','lon'),
            attrs={'units': 'degrees', 'gridtype': 'regular'}),
        'base_atmosphere_relative_vorticity': DataArray(
            vort_bar, 
            dims=('lat','lon'),
            attrs={'units': 's^-1', 'gridtype': 'regular'}),
        'perturbation_atmosphere_relative_vorticity': DataArray(
            vort_prime, 
            dims=('lat','lon'),
            attrs={'units': 's^-1', 'gridtype': 'regular'})
        }
    return ics

