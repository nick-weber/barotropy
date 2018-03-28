# -*- coding: utf-8 -*-

"""
Module containing the Prognostic object Dynamics and some helper functions
"""
import numpy as np
from sympl import Prognostic, get_constant
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
            'dims': ['lat', 'lon'],
            'units': 's^-1',
            'alias': 'vortp',
        },
        'base_atmosphere_relative_vorticity': {
            'dims': ['lat', 'lon'],
            'units': 's^-1',
            'alias': 'vortb',
        },
        'latitude': {
            'dims': ['lat', 'lon'],
            'units': 'radians',
        },
        'longitude': {
            'dims': ['lat', 'lon'],
            'units': 'radians',
        }
    }

    # DIAGS: u (mean & pert), v (mean & pert), and psi (mean & pert)
    diagnostic_properties = {
        'perturbation_eastward_wind': {
            'units': 'm s^-1',
            'alias': 'up'
        },
        'base_eastward_wind': {
            'units': 'm s^-1',
            'alias': 'ub'
        },
        'perturbation_northward_wind': {
            'units': 'm s^-1',
            'alias': 'vp'
        },
        'base_northward_wind': {
            'units': 'm s^-1',
            'alias': 'vb'
        },
        'perturbation_atmosphere_horizontal_streamfunction': {
            'units': 'm^2 s^-1',
            'alias': 'psip',
        },
        'base_atmosphere_horizontal_streamfunction': {
            'units': 'm^2 s^-1',
            'alias': 'psib',
        },
    }

    # TENDENCIES: vorticity (prime only)
    tendency_properties = {
        'perturbation_atmosphere_relative_vorticity': {
            'units': 's^-2'
        }
    }

    def __init__(self, ntrunc=21):
        self._ntrunc = ntrunc

    def array_call(self, state):
        """
        Calculates the vorticity tendency from the current state using
        the barotropic vorticity equation.

        Args
        ----
        state : dict
            A dictionary of numpy arrays containing the model state.

        Returns
        -------
        tendencies : dict
            A single-item dictionary containing the vorticity
            tendency numpy array.
        diagnostics : dict
            A dictionary of numpy arrays containing the diagnostics
            that were needed to compute the vorticity tendency.
        """

        # Get numpy arrays with specifications from input_properties
        vortp = state['vortp']
        vortb = state['vortb']
        theta = state['latitude']
        lamb = state['longitude']

        # Compute diagnostics (streamfunction) for tendency calculation
        # using spherical harmonics
        gridtype = state['longitude'].gridtype
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

        # Here we actually compute vorticity tendency
        # Compute tendency with beta as only forcing
        f = 2 * Omega * np.sin(theta)
        beta = s.getgrad(s.grdtospec(f, ntrunc=self._ntrunc))[1]
        dvort_dx = s.getgrad(vortp_spec + vortb_spec)[0]
        dvort_dy = s.getgrad(vortp_spec + vortb_spec)[1]
        vort_tend = - (vp+vb) * beta - (up+ub) * dvort_dx - (vp+vb) * dvort_dy
        tendencies = {
            'vortp': vort_tend,
        }

        # Collect the diagnostics into a dictionary
        diagnostics = {'up': up, 'vp': vp, 'ub': ub, 'vb': vb,
                       'psip': psip, 'psib': psib}

        return tendencies, diagnostics
