# -*- coding: utf-8 -*-

"""
Module containing the Prognostic object Dynamics and some helper functions
"""
import numpy as np
from sympl import (
    Prognostic, get_constant, get_numpy_arrays_with_properties,
    restore_data_arrays_with_properties
)
import spharm

Re = get_constant('planetary_radius', 'm')
Omega = get_constant('planetary_rotation_rate', 's^-1')


class LinearizedDynamics(Prognostic):
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
        }
    }

    # DIAGS: u (mean & pert), v (mean & pert), and psi (mean & pert)
    diagnostic_properties = {
        'perturbation_eastward_wind': {
            #'dims': ['lat', 'lon'],
            'units': 'm s^-1',
            'alias': 'up',
            'dims_like' : 'latitude'
        },
        'base_eastward_wind': {
            #'dims': ['lat', 'lon'],
            'units': 'm s^-1',
            'alias': 'ub',
            'dims_like' : 'latitude'
        },
        'perturbation_northward_wind': {
            #'dims': ['lat', 'lon'],
            'units': 'm s^-1',
            'alias': 'vp',
            'dims_like' : 'latitude'
        },
        'base_northward_wind': {
            #'dims': ['lat', 'lon'],
            'units': 'm s^-1',
            'alias': 'vb',
            'dims_like' : 'latitude'
        },
        'perturbation_atmosphere_horizontal_streamfunction': {
            #'dims': ['lat', 'lon'],
            'units': 'm^2 s^-1',
            'alias': 'psip',
            'dims_like' : 'latitude'
        },
        'base_atmosphere_horizontal_streamfunction': {
            #'dims': ['lat', 'lon'],
            'units': 'm^2 s^-1',
            'alias': 'psib',
            'dims_like' : 'latitude'
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

    def __call__(self, state):
        """
        Gets tendencies and diagnostics from the passed model state.
        Copied from sympl develop branch (to-be v0.3.3), ignoring checks.

        Args
        ----
        state : dict
            A model state dictionary.
        Returns
        -------
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second at the time of the input state.
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state.
        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the Prognostic instance.
        """
        raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
        raw_state['time'] = state['time']
        raw_tendencies, raw_diagnostics = self.array_call(raw_state)
        tendencies = restore_data_arrays_with_properties(
            raw_tendencies, self.tendency_properties,
            state, self.input_properties)
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics, self.diagnostic_properties,
            state, self.input_properties)
        return tendencies, diagnostics

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

        # Get numpy arrays from state
        vortp = state['vortp']
        vortb = state['vortb']
        theta = state['latitude']

        # Compute diagnostics (streamfunction) for tendency calculation
        # using spherical harmonics
        s = spharm.Spharmt(theta.shape[1], theta.shape[0], rsphere=Re,
                           gridtype='gaussian', legfunc='computed')
        vortp_spec = s.grdtospec(vortp, ntrunc=self._ntrunc)
        vortb_spec = s.grdtospec(vortb, ntrunc=self._ntrunc)
        div_spec = np.zeros(vortb_spec.shape)  # Only want NON-DIVERGENT wind
        # Get the winds
        up, vp = s.getuv(vortp_spec, div_spec)
        ub, vb = s.getuv(vortb_spec, div_spec)
        # And now the streamfunction
        psip, _ = s.getpsichi(up, vp)
        psib, _ = s.getpsichi(ub, vb)

        # Here we actually compute vorticity tendency from the linearized equation
        f = 2 * Omega * np.sin(theta)
        beta = s.getgrad(s.grdtospec(f, ntrunc=self._ntrunc))[1]
        dvortp_dx = s.getgrad(vortp_spec)[0]
        dvortp_dy = s.getgrad(vortp_spec)[1]
        dvortb_dx = s.getgrad(vortb_spec)[0]
        dvortb_dy = s.getgrad(vortb_spec)[1]
        vort_tend = - (vp+vb) * beta - up * dvortb_dx - vp * dvortb_dy - \
            ub * dvortp_dx - vb * dvortp_dy

        tendencies = {'vortp': vort_tend}

        # Collect the diagnostics into a dictionary
        diagnostics = {'up': up, 'vp': vp, 'ub': ub, 'vb': vb,
                       'psip': psip, 'psib': psib}

        return tendencies, diagnostics


class NonlinearDynamics(Prognostic):
    """
    Prescribes vorticity tendency based on the full barotropic
    vorticity equation.
    """

    # INPUT: vortcity (mean & pert), latitude, longitude
    input_properties = {
        'atmosphere_relative_vorticity': {
            'dims': ['lat', 'lon'],
            'units': 's^-1',
            'alias': 'vort',
        },
        'latitude': {
            'dims': ['lat', 'lon'],
            'units': 'radians',
        }
    }

    # DIAGS: u (mean & pert), v (mean & pert), and psi (mean & pert)
    diagnostic_properties = {
        'eastward_wind': {
            #'dims': ['lat', 'lon'],
            'units': 'm s^-1',
            'alias': 'u',
            'dims_like': 'latitude'
        },
        'northward_wind': {
            #'dims': ['lat', 'lon'],
            'units': 'm s^-1',
            'alias': 'v',
            'dims_like': 'latitude'
        },
        'atmosphere_horizontal_streamfunction': {
            #'dims': ['lat', 'lon'],
            'units': 'm^2 s^-1',
            'alias': 'psi',
            'dims_like': 'latitude'
        }
    }

    # TENDENCIES: vorticity (prime only)
    tendency_properties = {
        'atmosphere_relative_vorticity': {
            'units': 's^-2',
            'dims_like': 'latitude'
        }
    }

    def __init__(self, ntrunc=21):
        self._ntrunc = ntrunc

    def __call__(self, state):
        """
        Gets tendencies and diagnostics from the passed model state.
        Copied from sympl develop branch (to-be v0.3.3), ignoring checks.

        Args
        ----
        state : dict
            A model state dictionary.
        Returns
        -------
        tendencies : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the time derivative of those
            quantities in units/second at the time of the input state.
        diagnostics : dict
            A dictionary whose keys are strings indicating
            state quantities and values are the value of those quantities
            at the time of the input state.
        Raises
        ------
        KeyError
            If a required quantity is missing from the state.
        InvalidStateError
            If state is not a valid input for the Prognostic instance.
        """
        raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
        raw_state['time'] = state['time']
        raw_tendencies, raw_diagnostics = self.array_call(raw_state)
        tendencies = restore_data_arrays_with_properties(
            raw_tendencies, self.tendency_properties,
            state, self.input_properties)
        # print(state.keys())
        # print(raw_state.keys())
        # print(raw_diagnostics.keys())
        # print(self.diagnostic_properties.keys())
        # print(self.input_properties.keys())
        # print('-----------')
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics, self.diagnostic_properties,
            state, self.input_properties)
        return tendencies, diagnostics

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

        # Get numpy arrays from state
        vort = state['vort']
        theta = state['latitude']

        # Compute diagnostics (streamfunction) for tendency calculation
        # using spherical harmonics
        s = spharm.Spharmt(theta.shape[1], theta.shape[0], rsphere=Re,
                           gridtype='gaussian', legfunc='computed')
        vort_spec = s.grdtospec(vort, ntrunc=self._ntrunc)
        div_spec = np.zeros(vort_spec.shape)  # Only want NON-DIVERGENT wind
        # Get the winds
        u, v = s.getuv(vort_spec, div_spec)
        # And now the streamfunction
        psi, _ = s.getpsichi(u, v)

        # Here we actually compute vorticity tendency
        # Compute tendency with beta as only forcing
        f = 2 * Omega * np.sin(theta)
        beta = s.getgrad(s.grdtospec(f, ntrunc=self._ntrunc))[1]
        dvort_dx = s.getgrad(vort_spec)[0]
        dvort_dy = s.getgrad(vort_spec)[1]
        vort_tend = - v * beta - u * dvort_dx - v * dvort_dy
        tendencies = {'vort': vort_tend}

        # Collect the diagnostics into a dictionary
        diagnostics = {'u': u, 'v': v, 'psi': psi}

        return tendencies, diagnostics
