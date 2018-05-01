# -*- coding: utf-8 -*-

"""Top-level package for Barotropy."""

from ._core.initialize import super_rotation, sinusoidal_perts_on_zonal_jet, from_u_and_v_winds
from ._core.dynamics import LinearizedDynamics, NonlinearDynamics
from ._core.diffusion import LinearizedDiffusion, NonlinearDiffusion
from ._core.forcing import Forcing
from ._core.damping import LinearizedDamping, NonlinearDamping
from ._core.util import (
    gaussian_latlon_grid, interp_to_gaussian, gaussian_blob_2d
)

from.plotting import debug_plots

__author__ = """Nick Weber"""
__email__ = 'njweber@uw.edu'
__version__ = '0.1.0'
__all__ = (
    super_rotation, sinusoidal_perts_on_zonal_jet, from_u_and_v_winds,
    LinearizedDynamics, NonlinearDynamics, LinearizedDiffusion, NonlinearDiffusion,
    Forcing, LinearizedDamping, NonlinearDamping, gaussian_latlon_grid,
    interp_to_gaussian, gaussian_blob_2d, debug_plots
)

