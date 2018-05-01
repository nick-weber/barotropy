# -*- coding: utf-8 -*-

"""
Module containing the Prognostic object Dynamics and some helper functions
"""
from sympl import Prognostic


class LinearizedDamping(Prognostic):
    """
    Prescribes vorticity tendency due to linear damping.
    """

    # INPUT: perturbation vortcity
    input_properties = {
        'perturbation_atmosphere_relative_vorticity': {
            'dims': ['lat', 'lon'],
            'units': 's^-1',
            'alias': 'vortp',
        }
    }

    # DIAGS: none
    diagnostic_properties = {}

    # TENDENCIES: perturbation vorticity
    tendency_properties = {
        'perturbation_atmosphere_relative_vorticity': {
            'units': 's^-2'
        }
    }

    def __init__(self, tau=14.7, **kwargs):
        """
        Args
        ----
        tau : float
            Linear damping timescale in days.
        """
        self._tau = tau * 3600. * 24.  # days --> seconds
        if 'name' not in kwargs.keys() or kwargs['name'] is None:
            kwargs['name'] = 'damping'
        super(LinearizedDamping, self).__init__(**kwargs)

    def array_call(self, state):
        """
        Calculates the linear damping vorticity tendency from the current state.

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
            An empty dictionary.
        """

        # Get numpy arrays with specifications from input_properties
        vortp = state['vortp']

        # Compute the linear damping tendency
        vort_tend = - vortp / self._tau
        tendencies = {'vortp': vort_tend}

        diagnostics = {}

        return tendencies, diagnostics


class NonlinearDamping(Prognostic):
    """
    Prescribes vorticity tendency due to linear damping.
    """

    # INPUT: perturbation vortcity
    input_properties = {
        'atmosphere_relative_vorticity': {
            'dims': ['lat', 'lon'],
            'units': 's^-1',
            'alias': 'vort',
        }
    }

    # DIAGS: none
    diagnostic_properties = {}

    # TENDENCIES: perturbation vorticity
    tendency_properties = {
        'atmosphere_relative_vorticity': {
            'units': 's^-2'
        }
    }

    def __init__(self, tau=14.7, **kwargs):
        """
        Args
        ----
        tau : float
            Linear damping timescale in days.
        """
        self._tau = tau * 3600. * 24.  # days --> seconds
        if 'name' not in kwargs.keys() or kwargs['name'] is None:
            kwargs['name'] = 'damping'
        super(NonlinearDamping, self).__init__(**kwargs)

    def array_call(self, state):
        """
        Calculates the linear damping vorticity tendency from the current state.

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
            An empty dictionary.
        """

        # Get numpy arrays with specifications from input_properties
        vort = state['vort']

        # Compute the linear damping tendency
        vort_tend = - vort / self._tau
        tendencies = {'vort': vort_tend}

        diagnostics = {}

        return tendencies, diagnostics
