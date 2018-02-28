# -*- coding: utf-8 -*-

"""
Module containing the Prognostic object Dynamics and some helper functions
"""
import numpy as np
from sympl import (Prognostic, get_numpy_arrays_with_properties,
                   restore_data_arrays_with_properties)
import spharm
import namelist as NL

class Dynamics(Prognostic):
    """
    Prescribes vorticity tendency based on the linearized barotropic
    vorticity equation.
    """

    # INPUT: vortcity (mean & pert), latitude, longitude
    input_properties = {
        'perturbation_atmosphere_relative_vorticity': { 
            'dims': ['y','x'], 
            'units': 's^-1',
            'alias': 'vortp',
        },
        'base_atmosphere_relative_vorticity': { 
            'dims': ['y','x'], 
            'units': 's^-1',
            'alias': 'vortb',
        },
        'grid_latitude': {
            'dims': ['y', 'x'],
            'units': 'radians',
        },
        'grid_longitude': {
            'dims': ['y', 'x'],
            'units': 'radians',
        }
    }

    # DIAGS: u (mean & pert), v (mean & pert), and psi (mean & pert)
    diagnostic_properties = {
        'perturbation_eastward_wind': {
            'dims_like': 'grid_latitude',
            'units': 'm s^-1',
            'alias': 'up'
        },
        'base_eastward_wind': {
            'dims_like': 'grid_latitude',
            'units': 'm s^-1',
            'alias': 'ub'
        },
        'perturbation_northward_wind': {
            'dims_like': 'grid_latitude',
            'units': 'm s^-1',
            'alias': 'vp'
        },
        'base_northward_wind': {
            'dims_like': 'grid_latitude',
            'units': 'm s^-1',
            'alias': 'vb'
        },
        'perturbation_atmosphere_horizontal_streamfunction': { 
            'dims_like': 'grid_latitude',
            'units': 'm^2 s^-1',
            'alias': 'psip',
        },
        'base_atmosphere_horizontal_streamfunction': { 
            'dims_like': 'grid_latitude',
            'units': 'm^2 s^-1',
            'alias': 'psib',
        },
    }

    # TENDENCIES: vorticity (prime only)
    tendency_properties = {
        'perturbation_atmosphere_relative_vorticity': {
            'dims_like': 'grid_latitude',
            'units': 's^-2',
        }
    }

    def __init__(self, ntrunc=21, omega=7.292E-5):
        self._ntrunc = ntrunc
        self._omega = omega

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
        theta = raw_arrays['grid_latitude']
        lamb = raw_arrays['grid_longitude']
        
        # Compute diagnostics (streamfunction) for tendency calculation
        # using spherical harmonics
        gridtype = state['grid_longitude'].gridtype
        s = spharm.Spharmt(lamb.shape[1], lamb.shape[0], rsphere=NL.Re,
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
        
        # Compute dtheta and dlamba for the derivatives
        dlamb = np.gradient(lamb)[1]
        dtheta = np.gradient(theta)[0]
        
        # Here we actually compute vorticity tendency
        # Compute tendency with beta as only forcing
        vort_tend = -2. * self._omega/(NL.Re**2) * d_dlamb(psip + psib, dlamb) - \
            Jacobian(psip+psib, vortp+vortb, theta, dtheta, dlamb)
        raw_tendencies = {
            'vortp': vort_tend,
        }
        #######
        print('--- vorticity ---')
        print('min:', np.min(vortb+vortp))
        print('max:', np.max(vortb+vortp))
        print('--- tendency ---')
        print('min:', np.min(vort_tend))
        print('max:', np.max(vort_tend))
        #######
        
        # Collect the diagnostics into a dictionary
        raw_diagnostics = {'up': up, 'vp': vp, 'ub': ub, 'vb':vb,
                           'psip': psip, 'psib': psib}
        
        # Now we re-format the data in a way the host model can use
        tendencies = restore_data_arrays_with_properties(
            raw_tendencies, self.tendency_properties,
            state, self.input_properties)
        diagnostics = restore_data_arrays_with_properties(
            raw_diagnostics, self.diagnostic_properties,
            state, self.input_properties)
        
        ### TODO: this is not needed when TendencyInDiagnosticsWrapper() is fixed
        diagnostic_name = 'tendency_of_{}_due_to_dynamics'.format(
            list(tendencies.keys())[0])
        diagnostics[diagnostic_name] = list(tendencies.values())[0]
        return tendencies, diagnostics
        
        
def d_dlamb(field,dlamb):
    """ Finds a finite-difference approximation to gradient in
    the lambda (longitude) direction"""
    out = np.divide(np.gradient(field)[1],dlamb) 
    return out

def d_dtheta(field,dtheta):
    """ Finds a finite-difference approximation to gradient in
    the theta (latitude) direction """
    out = np.divide(np.gradient(field)[0],dtheta)
    return out

def Jacobian(A,B,theta,dtheta,dlamb):
    """ Returns the Jacobian of two fields """
    term1 = d_dlamb(A,dlamb) * d_dtheta(B,dtheta)
    term2 = d_dlamb(B,dlamb) * d_dtheta(A,dtheta)
    return 1./(NL.Re**2 * np.cos(theta)) * (term1 - term2)