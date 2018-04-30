# -*- coding: utf-8 -*-

from barotropy import dynamics, diffusion, initialize
from barotropy.plotting.debug_plots import fourpanel, fourpanel_polar
from sympl import (Leapfrog, PlotFunctionMonitor, NetCDFMonitor,
                   get_component_aliases, get_constant)
from datetime import timedelta
import re
import os
from time import time

Re = get_constant('planetary_radius', 'm')
Omega = get_constant('planetary_rotation_rate', 's^-1')


# ============ Adjustable Variables ============
# Integration Options
dt = timedelta(minutes=30)  # timestep
duration = '20_00:00'       # run duration ('<days>_<hours>:<mins>')
linearized = False          # run model using linearized state/equations?
ncout_freq = 6              # netcdf write frequency (hours)
plot_freq = 6               # plot Monitor call frequency (hours)
ntrunc = 42                 # triangular truncation for spharm (e.g., 21 --> T21)

# Diffusion Options
diff_on = True              # Use diffusion?
k = 2.338e16                # Diffusion coefficient for del^4 hyperdiffusion

# Forcing Options
forcing_on = False          # Apply vort. tendency forcing?
damp_ts = 14.7              # Damping timescale (in days)

# I/O Options
ncoutfile = os.path.join(os.getcwd(), 'gfdl_test.nc')
append_nc = False
# ==============================================

start = time()

# Get the initial state
state = initialize.sinusoidal_perts_on_zonal_jet(linearized=linearized)

# Set up the Timestepper with the desired Prognostics
if linearized:
    dynamics_prog = dynamics.LinearizedDynamics(ntrunc=ntrunc, tendencies_in_diagnostics=True)
    diffusion_prog = diffusion.LinearizedDiffusion(k=k, ntrunc=ntrunc, tendencies_in_diagnostics=True)
else:
    dynamics_prog = dynamics.NonlinearDynamics(ntrunc=ntrunc, tendencies_in_diagnostics=True)
    diffusion_prog = diffusion.NonlinearDiffusion(k=k, ntrunc=ntrunc, tendencies_in_diagnostics=True)

prognostics = [dynamics_prog]
if diff_on:
    prognostics.append(diffusion_prog)

stepper = Leapfrog(*prognostics)

# Create Monitors for plotting & storing data
plt_monitor = PlotFunctionMonitor(fourpanel)
if os.path.isfile(ncoutfile) and not append_nc:
    os.remove(ncoutfile)

aliases = get_component_aliases(*prognostics)
nc_monitor = NetCDFMonitor(ncoutfile, write_on_store=True, aliases=aliases)

# Figure out the end date of this run
d, h, m = re.split('[_:]', duration)
end_date = state['time'] + timedelta(days=int(d), hours=int(h), minutes=int(m))

# Begin the integration loop
idate = state['time']
while state['time'] <= end_date:
    # Get the state at the next timestep using our Timestepper
    dynamics_diagnostics, next_state = stepper(state, dt)

    # Add any calculated diagnostics to our current state
    state.update(dynamics_diagnostics)

    # if diff_on:
    #     diffusion_diagnostics, next_state = diff_stepper(state, dt)
    #     state.update(diffusion_diagnostics)
    # else:
    #     next_state = post_dynamics_state

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

print('TOTAL INTEGRATION TIME: {:.02f} min\n'.format((time()-start)/60.))
