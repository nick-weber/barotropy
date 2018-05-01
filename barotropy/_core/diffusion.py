# -*- coding: utf-8 -*-

"""
Module containing the Prognostic object Diffusion and some helper functions
"""
from sympl import Prognostic, get_numpy_arrays_with_properties, restore_data_arrays_with_properties
import spharm


class LinearizedDiffusion(Prognostic):
    """
    Prescribes vorticity tendency based due to del^4 hyperdiffusion.
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
        }
    }

    # DIAGS: none
    diagnostic_properties = {}

    # TENDENCIES: vorticity (prime only)
    tendency_properties = {
        'perturbation_atmosphere_relative_vorticity': {
            'units': 's^-2',
        }
    }

    def __init__(self, ntrunc=21, k=2.338e16):
        self._ntrunc = ntrunc
        self._k = k

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
        Calculates the vorticity tendency from the current state using:
        diffusion = k * del^4(vorticity)

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
        vortb = state['vortb']

        # Approximate del^4(total vorticity)
        del4vort = compute_del4vort(vortp+vortb, self._ntrunc)
        tendencies = {'vortp': -self._k * del4vort}

        diagnostics = {}

        return tendencies, diagnostics


class NonlinearDiffusion(Prognostic):
    """
    Prescribes vorticity tendency based due to del^4 hyperdiffusion.
    """

    # INPUT: vortcity (mean & pert), latitude, longitude
    input_properties = {
        'atmosphere_relative_vorticity': {
            'dims': ['lat', 'lon'],
            'units': 's^-1',
            'alias': 'vort',
        }
    }

    # DIAGS: none
    diagnostic_properties = {}

    # TENDENCIES: vorticity (prime only)
    tendency_properties = {
        'atmosphere_relative_vorticity': {
            'units': 's^-2',
        }
    }

    def __init__(self, ntrunc=21, k=2.338e16):
        self._ntrunc = ntrunc
        self._k = k

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
        Calculates the vorticity tendency from the current state using:
        diffusion = k * del^4(vorticity)

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

        # Approximate del^4(total vorticity)
        del4vort = compute_del4vort(vort, self._ntrunc)
        tendencies = {'vort': -self._k * del4vort}

        diagnostics = {}

        return tendencies, diagnostics


def compute_del4vort(vort, ntrunc):
    # Compute del^4(vorticity) with spherical harmonics
    # Approximating del^4 as: d4_dx4 + d4_dy4 + 2 * (d2_dx2 * d2_dy2)
    s = spharm.Spharmt(vort.shape[1], vort.shape[0], rsphere=6378100.,
                       gridtype='gaussian', legfunc='computed')
    vspec = s.grdtospec(vort, ntrunc=ntrunc)
    # First order
    dvort_dx, dvort_dy = s.getgrad(vspec)
    # Second order
    d2vort_dx2, _ = s.getgrad(s.grdtospec(dvort_dx, ntrunc=ntrunc))
    _, d2vort_dy2 = s.getgrad(s.grdtospec(dvort_dy, ntrunc=ntrunc))
    # Fourth order
    d4vort_dx4, _ = s.getgrad(s.grdtospec(s.getgrad(s.grdtospec(d2vort_dx2,
                                                                ntrunc=ntrunc))[0], ntrunc=ntrunc))
    _, d4vort_dy4 = s.getgrad(s.grdtospec(s.getgrad(s.grdtospec(d2vort_dy2,
                                                                ntrunc=ntrunc))[1], ntrunc=ntrunc))
    # Put it all together to approximate del^4
    del4vort = d4vort_dx4 + d4vort_dy4 + (2 * d2vort_dx2 * d2vort_dy2)
    return del4vort
