# -*- coding: utf-8 -*-

from initialize import sinusoidal_NH
from dynamics import Dynamics
from diffusion import Diffusion
from sympl import Leapfrog, TendencyInDiagnosticsWrapper, NetCDFMonitor
from datetime import timedelta
import namelist as NL
import os

state = sinusoidal_NH()
# WILL BE ABLE TO DO THIS in next sympl release:
# stepper = Leapfrog([TendencyInDiagnosticsWrapper(Dynamics(), 'dynamics')])
stepper = Leapfrog([Dynamics()])#, Diffusion()])

ncoutfile = os.path.join(os.getcwd(), 'test.nc')
monitor = NetCDFMonitor(ncoutfile, write_on_store=True)

timestep = timedelta(minutes=NL.dt)

for t in range(4*24):
    diagnostics, next_state = stepper(state, timestep)

    state.update(diagnostics)
    
    if t%4==0:  # save to netcdf hourly
        print(state['time'])
        monitor.store(state)

    next_state['time'] = state['time'] + timestep
    state = next_state
    
print('Finished!')