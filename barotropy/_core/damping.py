# -*- coding: utf-8 -*-

"""
Module containing the Prognostic object Dynamics and some helper functions
"""
from sympl import Prognostic, get_numpy_arrays_with_properties, restore_data_arrays_with_properties


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

    def __init__(self, tau=14.7):
        """
        Args
        ----
        tau : float
            Linear damping timescale in days.
        """
        self._tau = tau * 3600. * 24.  # days --> seconds

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

    def __init__(self, tau=14.7):
        """
        Args
        ----
        tau : float
            Linear damping timescale in days.
        """
        self._tau = tau * 3600. * 24.  # days --> seconds

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
