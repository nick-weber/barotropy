# -*- coding: utf-8 -*-

"""
Module containing the Prognostic object Diffusion and some helper functions
"""
from sympl import Prognostic
import spharm


class Diffusion(Prognostic):
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

    def __init__(self, ntrunc=21, k=2.338e16, tendencies_in_diagnostics=False, name=None):
        self._ntrunc = ntrunc
        self._k = k
        if name is None:
            name = 'dynamics'
        super(Diffusion, self).__init__(
            tendencies_in_diagnostics=tendencies_in_diagnostics, name=name
        )

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

        # Compute del^4(vorticity) with spherical harmonics
        # Approximating del^4 as: d4_dx4 + d4_dy4 + 2 * (d2_dx2 * d2_dy2)
        s = spharm.Spharmt(vortp.shape[1], vortp.shape[0], rsphere=6378100.,
                           gridtype='regular', legfunc='computed')
        vspec = s.grdtospec(vortp+vortb, ntrunc=self._ntrunc)
        # First order
        dvort_dx, dvort_dy = s.getgrad(vspec)
        # Second order
        d2vort_dx2, _ = s.getgrad(s.grdtospec(dvort_dx, ntrunc=self._ntrunc))
        _, d2vort_dy2 = s.getgrad(s.grdtospec(dvort_dy, ntrunc=self._ntrunc))
        # Fourth order
        d4vort_dx4, _ = s.getgrad(s.grdtospec(s.getgrad(s.grdtospec(d2vort_dx2,
                                  ntrunc=self._ntrunc))[0], ntrunc=self._ntrunc))
        _, d4vort_dy4 = s.getgrad(s.grdtospec(s.getgrad(s.grdtospec(d2vort_dy2,
                                  ntrunc=self._ntrunc))[1], ntrunc=self._ntrunc))
        # Put it all together to approximate del^4
        del4vort = d4vort_dx4 + d4vort_dy4 + (2 * d2vort_dx2 * d2vort_dy2)
        tendencies = {'vortp': -self._k * del4vort}

        diagnostics = {}

        return tendencies, diagnostics
