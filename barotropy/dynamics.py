# -*- coding: utf-8 -*-

"""
Module containing the Prognostic object Dynamics and some helper functions
"""
import numpy as np
from sympl import (Prognostic, get_numpy_arrays_with_properties,
                   restore_data_arrays_with_properties, get_constant)
import spharm

Re = get_constant('planetary_radius', 'm')
Omega = get_constant('planetary_rotation_rate', 's^-1')


class Dynamics(Prognostic):
    """
    Prescribes vorticity tendency based on the linearized barotropic
    vorticity equation.
    """

    # INPUT: vortcity (mean & pert), latitude, longitude
    input_properties = {
        'perturbation_atmosphere_relative_vorticity': {
            'dims': ['y', 'x'],
            'units': 's^-1',
            'alias': 'vortp',
        },
        'base_atmosphere_relative_vorticity': {
            'dims': ['y', 'x'],
            'units': 's^-1',
            'alias': 'vortb',
        },
        'lat': {
            'dims': ['y', 'x'],
            'units': 'radians',
        },
        'lon': {
            'dims': ['y', 'x'],
            'units': 'radians',
        }
    }

    # DIAGS: u (mean & pert), v (mean & pert), and psi (mean & pert)
    diagnostic_properties = {
        'perturbation_eastward_wind': {
            'dims_like': 'lat',
            'units': 'm s^-1',
            'alias': 'up'
        },
        'base_eastward_wind': {
            'dims_like': 'lat',
            'units': 'm s^-1',
            'alias': 'ub'
        },
        'perturbation_northward_wind': {
            'dims_like': 'lat',
            'units': 'm s^-1',
            'alias': 'vp'
        },
        'base_northward_wind': {
            'dims_like': 'lat',
            'units': 'm s^-1',
            'alias': 'vb'
        },
        'perturbation_atmosphere_horizontal_streamfunction': {
            'dims_like': 'lat',
            'units': 'm^2 s^-1',
            'alias': 'psip',
        },
        'base_atmosphere_horizontal_streamfunction': {
            'dims_like': 'lat',
            'units': 'm^2 s^-1',
            'alias': 'psib',
        },
    }

    # TENDENCIES: vorticity (prime only)
    tendency_properties = {
        'perturbation_atmosphere_relative_vorticity': {
            'dims_like': 'lat',
            'units': 's^-2'
        }
    }

    def __init__(self, ntrunc=21):
        self._ntrunc = ntrunc

    def __call__(self, state):
        """
        Calculates the vorticity tendency from the current state using
        the barotropic vorticity equation.

        Args
        ----
        state : dict
            A dictionary of DataArrays containing the model state.

        Returns
        -------
        tendencies : dict
            A single-item dictionary containing the vorticity
            tendency DataArray.
        diagnostics : dict
            A dictionary of DataArrays containing the diagnostics
            that were needed to compute the vorticity tendency.
        """

        # Get numpy arrays with specifications from input_properties
        raw_arrays = get_numpy_arrays_with_properties(
            state, self.input_properties)
        vortp = raw_arrays['vortp']
        vortb = raw_arrays['vortb']
        theta = raw_arrays['lat']
        lamb = raw_arrays['lon']

        # Compute diagnostics (streamfunction) for tendency calculation
        # using spherical harmonics
        gridtype = state['lon'].gridtype
        s = spharm.Spharmt(lamb.shape[1], lamb.shape[0], rsphere=Re,
                           gridtype=gridtype, legfunc='computed')
        vortp_spec = s.grdtospec(vortp, ntrunc=self._ntrunc)
        vortb_spec = s.grdtospec(vortb, ntrunc=self._ntrunc)
        div_spec = np.zeros(vortb_spec.shape)  # Only want NON-DIVERGENT wind
        # Get the winds
        up, vp = s.getuv(vortp_spec, div_spec)
        ub, vb = s.getuv(vortb_spec, div_spec)
        # And now the streamfunction
        psip, _ = s.getpsichi(up, vp)
        psib, _ = s.getpsichi(ub, vb)
        # # And the post-truncation vorticity
        # vortp = s.spectogrd(vortp_spec)
        # # ^ no need to do the base state, as that is probably already smooth

        # Compute dtheta and dlamba for the derivatives
        dlamb = np.gradient(lamb)[1]
        dtheta = np.gradient(theta)[0]

        # Here we actually compute vorticity tendency
        # Compute tendency with beta as only forcing
        f = 2 * Omega * np.sin(theta)
        beta = s.getgrad(s.grdtospec(f, ntrunc=self._ntrunc))[1]
        dvort_dx = s.getgrad(vortp_spec + vortb_spec)[0]
        dvort_dy = s.getgrad(vortp_spec + vortb_spec)[1]
        vort_tend = - (vp+vb) * beta - (up+ub) * dvort_dx - (vp+vb) * dvort_dy
        raw_tendencies = {
            'vortp': vort_tend,
        }

        # Collect the diagnostics into a dictionary
        raw_diagnostics = {'up': up, 'vp': vp, 'ub': ub, 'vb': vb,
                           'psip': psip, 'psib': psib}

        # Now we re-format the data in a way the host model can use
        tendencies = restore_data_arrays_with_properties(
            raw_tendencies, self.tendency_properties,
            state, self.input_properties)
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics, self.diagnostic_properties,
            state, self.input_properties)

        return tendencies, diagnostics


def d_dlamb(field, dlamb):
    """ Finds a finite-difference approximation to gradient in
    the lambda (longitude) direction"""
    # Buffer the data in the lon direction to make it x-periodic
    rbound = field[:, :2]
    lbound = field[:, -2:]
    buffered_field = np.hstack([lbound, field, rbound])
    # Do the same to the longitudes
    buffered_dlamb = np.hstack([dlamb[:, -2:], dlamb, dlamb[:, :2]])
    # Compute the gradient and trim the buffer off
    out = np.divide(np.gradient(buffered_field)[1], buffered_dlamb)
    return out[:, 2:-2]


def d_dtheta(field, dtheta):
    """ Finds a finite-difference approximation to gradient in
    the theta (latitude) direction """
    out = np.divide(np.gradient(field)[0], dtheta)
    return out


def jacobian(a, b, theta, dtheta, dlamb):
    """ Returns the Jacobian of two fields """
    term1 = d_dlamb(a, dlamb) * d_dtheta(b, dtheta)
    term2 = d_dlamb(b, dlamb) * d_dtheta(a, dtheta)
    return 1./(Re**2 * np.cos(theta)) * (term1 - term2)
