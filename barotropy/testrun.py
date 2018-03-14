# -*- coding: utf-8 -*-

from barotropy.initialize import from_u_and_v_winds
from barotropy.dynamics import Dynamics
from barotropy.diffusion import Diffusion
from barotropy.forcing import Forcing
from barotropy.plotting.debug_plots import fourpanel
from sympl import (Leapfrog, TendencyInDiagnosticsWrapper,
                   PlotFunctionMonitor, NetCDFMonitor,
                   get_component_aliases)
from datetime import timedelta
import re
import os
from netCDF4 import Dataset
import numpy as np

# import traceback
# import warnings
# import sys
#
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
# warnings.showwarning = warn_with_traceback

# ============ Adjustable Variables ============
# Integration Options
dt = timedelta(minutes=15)  # timestep
duration = '5_00:00'       # run duration ('<days>_<hours>:<mins>')t
ncout_freq = 6              # netcdf write frequency (hours)
plot_freq = 6               # plot Monitor call frequency (hours)
ntrunc = 21                 # triangular truncation for spharm (e.g., 21 --> T21)

# Diffusion Options
k = 2.338e16                # Diffusion coefficient for del^4 hyperdiffusion

# I/O Options
ncoutfile = os.path.join(os.getcwd(), 'test.nc')
append_nc = False
# ==============================================


# Get the initial state
with Dataset('wnd200_DJF.nc', 'r') as ncdata:
    lats = ncdata.variables['latitude'][:]
    lons = ncdata.variables['longitude'][:]
    ubar = ncdata.variables['U200'][:,:]
    vbar = ncdata.variables['V200'][:,:]
uprime = np.zeros(ubar.shape)
vprime = np.zeros(vbar.shape)
state = from_u_and_v_winds(lats, lons, ubar, vbar, uprime, vprime,
                           idate=None, lats_increasing=True)

# Get our Gaussian RWS forcing
rwsforcing = Forcing.gaussian_tendencies(state['lat'].values, state['lon'].values)

# Set up the Timestepper with the desired Prognostics
prognostics = [TendencyInDiagnosticsWrapper(Dynamics(ntrunc=ntrunc), 'dynamics'),
               TendencyInDiagnosticsWrapper(Diffusion(ntrunc=ntrunc), 'diffusion'),
               TendencyInDiagnosticsWrapper(rwsforcing, 'forcing')]
stepper = Leapfrog(prognostics)

# Create Monitors for plotting & storing data
plt_monitor = PlotFunctionMonitor(fourpanel)
if os.path.isfile(ncoutfile) and not append_nc:
    os.remove(ncoutfile)
nc_monitor = NetCDFMonitor(ncoutfile, write_on_store=True,
    aliases=get_component_aliases(prognostics))

# Figure out the end date of this run
d, h, m = re.split('[_:]', duration)
end_date = state['time'] + timedelta(days=int(d), hours=int(h), minutes=int(m))

# Begin the integration loop
idate = state['time']
while state['time'] <= end_date:

    # Get the state at the next timestep using our Timestepper
    diagnostics, next_state = stepper(state, dt)

    # Add any calculated diagnostics to our current state
    state.update(diagnostics)

    # Write state to netCDF every <ncout_freq> hours
    fhour = (state['time'] - idate).days*24 + (state['time'] - idate).seconds/3600
    if fhour % ncout_freq == 0:
        print(state['time'])
        nc_monitor.store(state)

    # Make plot(s) every <plot_freq> hours
    if fhour % plot_freq == 0:
        plt_monitor.store(state)

    # Advance the state to the next timestep
    next_state['time'] = state['time'] + dt
    state = next_state

print('Finished!')
