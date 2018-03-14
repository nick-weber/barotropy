# -*- coding: utf-8 -*-

"""
Module containing the Prognostic object Diffusion and some helper functions
"""
import numpy as np
from sympl import (Prognostic, get_numpy_arrays_with_properties,
                   restore_data_arrays_with_properties)
import spharm


class Diffusion(Prognostic):
    """
    Prescribes vorticity tendency based due to del^4 hyperdiffusion.
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

    # DIAGS: none
    diagnostic_properties = {}

    # TENDENCIES: vorticity (prime only)
    tendency_properties = {
        'perturbation_atmosphere_relative_vorticity': {
            'dims_like': 'lat',
            'units': 's^-2',
        }
    }

    def __init__(self, ntrunc=21, k=2.338e16):
        self._ntrunc = ntrunc
        self._k = k

    def __call__(self, state):
        """
        Calculates the vorticity tendency from the current state using:
        diffusion = k * del^4(vorticity)

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
            An empty dictionary.
        """

        # Get numpy arrays with specifications from input_properties
        raw_arrays = get_numpy_arrays_with_properties(
            state, self.input_properties)
        vortp = raw_arrays['vortp']
        vortb = raw_arrays['vortb']
        theta = raw_arrays['lat']
        lamb = raw_arrays['lon']

        # # Calculate the vorticity tendency due to diffusion
        # raw_tendencies = {
        #     'vortp': self._k * del4(vortp+vortb, theta, lamb),
        # }

        ### TRYING THINGS
        s = spharm.Spharmt(lamb.shape[1], lamb.shape[0], rsphere=6378100.,
            gridtype='regular', legfunc='computed')
        vspec = s.grdtospec(vortp+vortb, ntrunc=self._ntrunc)
        # FIRST ORDER
        dv_dx, dv_dy = s.getgrad(vspec)
        # SECOND ORDER
        d2v_dx2, _ = s.getgrad(s.grdtospec(dv_dx, ntrunc=self._ntrunc))
        _, d2v_dy2 = s.getgrad(s.grdtospec(dv_dy, ntrunc=self._ntrunc))
        # FOURTH ORDER
        d4v_dx4, _ = s.getgrad(s.grdtospec(s.getgrad(s.grdtospec(d2v_dx2,
            ntrunc=self._ntrunc))[0], ntrunc=self._ntrunc))
        _, d4v_dy4 = s.getgrad(s.grdtospec(s.getgrad(s.grdtospec(d2v_dy2,
            ntrunc=self._ntrunc))[1], ntrunc=self._ntrunc))
        # PUT IT ALL TOGETHER
        del4v = d4v_dx4 + d4v_dy4 + (2 * d2v_dx2 * d2v_dy2)
        raw_tendencies = {
            'vortp': -self._k * del4v,
        }

        # Now we re-format the data in a way the host model can use
        tendencies = restore_data_arrays_with_properties(
            raw_tendencies, self.tendency_properties,
            state, self.input_properties)
        diagnostics = {}

        return tendencies, diagnostics


def del4(data, theta, lamb, buflat=80.):
    """
    Applies the del^4 operator to 2D global data in x-y space.
    Uses the given lats/lons to compute spatial derivatives in meters.

    Args
    ----
    data : numpy array
        2D global field (e.g., vorticity); shape = (nlats, nlons)
    theta : numpy array
        2D latitude grid (in radians); same shape as data
    lamb : numpy array
        2D longitude grid (in radians); same shape as data
    buflat : float
        Latitude poleward of which zonal derivatives are set to zero.

    Returns
    -------
    del4data : numpy array
        output array; same shape as data
    """
    # Use the lat/lon mesh to calculate the distance (in meters)
    # between each point
    dy = 111000. * np.rad2deg(theta[1:,:]-theta[:-1,:])
    dx = np.cos(theta[:,1:]) * 111000. * np.rad2deg(lamb[:,1:]-lamb[:,:-1])

    # Calculate 2nd and 4th derivatives in x and y directions
    d2data_dy2 = second_derivative(data, dy, axis=0)
    d2data_dx2 = second_derivative(data, dx, axis=1)
    d4data_dy4 = fourth_derivative(data, dy, axis=0)
    d4data_dx4 = fourth_derivative(data, dx, axis=1)

    # Find how many lat points are poleward of <buflat> degrees; let's buffer
    # the x-derivatives with zeros here, since the very small dx values
    # make crazy derivatives
    nbuf = next((i for i,x in enumerate(np.rad2deg(theta[:,0])) if abs(x) < buflat))
    for derivative in [d2data_dx2, d4data_dx4]:
        derivative[:nbuf, :] = 0
        derivative[-nbuf:, :] = 0

    del4data = d4data_dy4 + d4data_dx4 + (2 * d2data_dy2 * d2data_dx2)

    # Use the above to calculate/return del^4(data)
    return del4data


def second_derivative(data, delta, axis=0):
    """
    Computes the second derivative of an Nd-array along the desired axis.

    Requires:
    data ---> N-dimensional numpy array
    delta --> float or 1-dimensional array/list (same length as desired <data> axis)
              indicating the distance between data points
    axis ---> desired axis to take the derivative along

    Returns:
    N-dimensional numpy array (same shape as <data>)
    """
    n = len(data.shape)

    # If <delta> is not an array (i.e., if the mesh is uniform),
    # create an array of deltas that is the same shape as <data>
    deltashape = list(data.shape)
    deltashape[axis] -= 1
    if type(delta) in [float, int, np.float64]:
        delta = np.ones(deltashape) * delta
    elif type(delta)==np.ndarray and len(np.shape(delta))==2:
        if np.shape(delta) != tuple(deltashape):
            print(np.shape(delta), tuple(deltashape))
            raise ValueError('input <delta> has invalid shape')
    else:
        raise ValueError('input <delta> should be value or 2D array')

    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)] * n
    slice1 = [slice(None)] * n
    slice2 = [slice(None)] * n
    delta_slice0 = [slice(None)] * n
    delta_slice1 = [slice(None)] * n

    # First handle centered case
    slice0[axis] = slice(None, -2)
    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(2, None)
    delta_slice0[axis] = slice(None, -1)
    delta_slice1[axis] = slice(1, None)
    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    center = 2 * (data[slice0] / (combined_delta * delta[delta_slice0]) -
                  data[slice1] / (delta[delta_slice0] * delta[delta_slice1]) +
                  data[slice2] / (combined_delta * delta[delta_slice1]))

    # Fill the left boundary (pad it with the edge value)
    slice0[axis] = slice(None,1)
    left = center[slice0].repeat(1, axis=axis)

    # Fill the right boundary (pad it with the edge value)
    slice0[axis] = slice(-1, None)
    right = center[slice0].repeat(1, axis=axis)

    return np.concatenate((left, center, right), axis=axis)




def fourth_derivative(data, delta, axis=0):
    """
    Computes the fourth derivative of Nd-array <data> along the desired axis.

    Requires:
    data ---> N-dimensional numpy array
    delta --> float or 1-dimensional array/list (same length as desired <data> axis)
              indicating the distance between data points
    axis ---> desired axis to take the derivative along

    Returns:
    N-dimensional numpy array (same shape as <data>)
    """
    n = len(data.shape)

    # If <delta> is not an array (i.e., if the mesh is uniform),
    # create an array of deltas that is the same shape as <data>
    deltashape = list(data.shape)
    deltashape[axis] -= 1
    if type(delta) in [float, int, np.float64]:
        delta = np.ones(deltashape) * delta
    elif type(delta)==np.ndarray and len(np.shape(delta))==2:
        if np.shape(delta) != tuple(deltashape):
            print(np.shape(delta), tuple(deltashape))
            raise ValueError('input <delta> has invalid shape')
    else:
        raise ValueError('input <delta> should be value or 2D array')


    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)] * n
    slice1 = [slice(None)] * n
    slice2 = [slice(None)] * n
    slice3 = [slice(None)] * n
    slice4 = [slice(None)] * n
    delta_slice0 = [slice(None)] * n
    delta_slice1 = [slice(None)] * n
    delta_slice2 = [slice(None)] * n
    delta_slice3 = [slice(None)] * n


    # First handle centered case
    slice0[axis] = slice(None, -4)
    slice1[axis] = slice(1, -3)
    slice2[axis] = slice(2, -2)
    slice3[axis] = slice(3, -1)
    slice4[axis] = slice(4, None)
    delta_slice0[axis] = slice(None, -3)
    delta_slice1[axis] = slice(1, -2)
    delta_slice2[axis] = slice(2, -1)
    delta_slice3[axis] = slice(3, None)
    center = f4(data[slice0], data[slice1], data[slice2], data[slice3],
                data[slice4], delta[delta_slice0], delta[delta_slice1],
                delta[delta_slice2], delta[delta_slice3])

    # Fill the left boundary (pad it with the edge value)
    slice0[axis] = slice(None,1)
    left = center[slice0].repeat(2, axis=axis)

    # Fill the right boundary (pad it with the edge value)
    slice0[axis] = slice(-1, None)
    right = center[slice0].repeat(2, axis=axis)

    return np.concatenate((left, center, right), axis=axis)




def f4(f0, f1, f2, f3, f4, d0, d1, d2, d3):
    """
    Computes the fourth derivative with the approximation:
    f^4(x) = [f(x-2) - 4*f(x-1) + 6*f(x) - 4*f(x+1) + f(x+2)] / hx^4
    (modified for non-uniform grid spacing)

    Requires:
    f0,f1,f2,f3,f4 -> values at the staggered locations (numerator)
    d0,d1,d2,d3 ----> distances (deltas) between the staggered locations (denominator)
    """
    d01 = d0 + d1
    d02 = d0 + d1 + d2
    d03 = d0 + d1 + d2 + d3
    d12 = d1 + d2
    d13 = d1 + d2 + d3
    d23 = d2 + d3
    return 24 * (f0/(d0*d01*d02*d03) - f1/(d0*d1*d12*d13) + f2/(d01*d1*d2*d23) -
                 f3/(d02*d12*d2*d3) + f4/(d03*d13*d23*d3))
