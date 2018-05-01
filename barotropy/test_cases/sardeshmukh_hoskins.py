# -*- coding: utf-8 -*-

from barotropy import (
    LinearizedDynamics, LinearizedDiffusion, LinearizedDamping, Forcing,
    NonlinearDynamics, NonlinearDiffusion, NonlinearDamping,
    super_rotation, debug_plots, gaussian_blob_2d
)
from sympl import (Leapfrog, PlotFunctionMonitor, NetCDFMonitor,
                   get_component_aliases, get_constant)
from datetime import timedelta
import re
import os
import numpy as np
from time import time
import spharm

Re = get_constant('planetary_radius', 'm')
Omega = get_constant('planetary_rotation_rate', 's^-1')


def main():
    # ============ Adjustable Variables ============
    # Integration Options
    dt = timedelta(minutes=15)  # timestep
    duration = '48_00:00'       # run duration ('<days>_<hours>:<mins>')t
    linearized = True
    ncout_freq = 6              # netcdf write frequency (hours)
    plot_freq = 6               # plot Monitor call frequency (hours)
    ntrunc = 42                 # triangular truncation for spharm (e.g., 21 --> T21)

    # Diffusion Options
    diff_on = True              # Use diffusion?
    k = 2.338e16                # Diffusion coefficient for del^4 hyperdiffusion

    # Forcing Options
    forcing_on = True           # Apply vort. tendency forcing?
    damp_ts = 14.7              # Damping timescale (in days)

    # I/O Options
    ncoutfile = os.path.join(os.getcwd(), 'sardeshmukh88.nc')
    append_nc = False           # Append to an existing netCDF file?
    # ==============================================

    start = time()

    # Get the initial state
    state = super_rotation(linearized=linearized)

    # Set up the Timestepper with the desired Prognostics
    if linearized:
        dynamics_prog = LinearizedDynamics(ntrunc=ntrunc, tendencies_in_diagnostics=True)
        diffusion_prog = LinearizedDiffusion(k=k, ntrunc=ntrunc, tendencies_in_diagnostics=True)
        damping_prog = LinearizedDamping(tau=damp_ts, tendencies_in_diagnostics=True)
    else:
        dynamics_prog = NonlinearDynamics(ntrunc=ntrunc, tendencies_in_diagnostics=True)
        diffusion_prog = NonlinearDiffusion(k=k, ntrunc=ntrunc, tendencies_in_diagnostics=True)
        damping_prog = NonlinearDamping(tau=damp_ts, tendencies_in_diagnostics=True)
    prognostics = [dynamics_prog]
    if diff_on:
        prognostics.append(diffusion_prog)
    if forcing_on:
        # Get our suptropical RWS forcing (from equatorial divergence)
        rws = rws_from_tropical_divergence(state)
        prognostics.append(Forcing.from_numpy_array(rws, linearized=linearized, tendencies_in_diagnostics=True))
        prognostics.append(damping_prog)
    stepper = Leapfrog(*prognostics)

    # Create Monitors for plotting & storing data
    plt_monitor = PlotFunctionMonitor(debug_plots.fourpanel)
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

    print('TOTAL INTEGRATION TIME: {:.02f} min\n'.format((time()-start)/60.))


def rws_from_tropical_divergence(state, center=(0., 145.), amp=6e-6, width=12):
    # Get desired state variables
    lats = state['latitude'].values
    lons = state['longitude'].values
    vort_bar = state['base_atmosphere_relative_vorticity'].values
    s = spharm.Spharmt(lats.shape[1], lons.shape[0], gridtype='regular', rsphere=Re)
    vortb_spec = s.grdtospec(vort_bar)
    ubar, vbar = s.getuv(vortb_spec, np.zeros(vortb_spec.shape))
    divergence = gaussian_blob_2d(lats, lons, center, width, amp)

    # Calculate the Rossby Wave Source
    # Term 1
    zetabar_spec, _ = s.getvrtdivspec(ubar, vbar)
    zetabar = s.spectogrd(zetabar_spec) + 2 * Omega * np.sin(np.deg2rad(lats))
    term1 = -zetabar * divergence
    # Term 2
    uchi, vchi = s.getuv(np.zeros(zetabar_spec.shape), s.grdtospec(divergence))
    dzeta_dx, dzeta_dy = s.getgrad(s.grdtospec(zetabar))
    term2 = - uchi * dzeta_dx - vchi * dzeta_dy
    rws = term1 + term2
    return rws


if __name__ == '__main__':
    main()
