# -*- coding: utf-8 -*-

"""
Module containing plotting functions that are designed for use with
Sympl's PlotFunctionMonitor to get a quick-and-dirty look at the
model output as it runs.
"""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from .utils import NonlinearCmap
# TODO: REMOVE THIS
import os


def pert_vort(fig, state):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title('perturbation vorticity [s$^{-1}$] -- ' +
                 '{:%Y-%m-%d %H:%M}'.format(state['time']))
    x = state['lon'].values
    y = state['lat'].values
    z = state['perturbation_atmosphere_relative_vorticity'].values
    im = ax.pcolormesh(x, y, z, vmin=-5e-5, vmax=5e-5, cmap=cm.get_cmap('RdBu_r'))
    fig.colorbar(im)


def fourpanel(fig, state):
    ax1 = fig.add_axes([0.05, 0.52, 0.40, 0.42])
    ax2 = fig.add_axes([0.47, 0.52, 0.40, 0.42])
    ax3 = fig.add_axes([0.05, 0.05, 0.40, 0.42])
    ax4 = fig.add_axes([0.47, 0.05, 0.40, 0.42])
    cax1 = fig.add_axes([0.89, 0.52, 0.02, 0.42])
    cax2 = fig.add_axes([0.89, 0.05, 0.02, 0.42])

    vortlevs = np.array([-10, -7.5, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 7.5, 10])
    tendlevs = np.array([-15, -5, -1, -0.5, -0.2, -0.05, -0.01, 0.01, 0.05, 0.2, 0.5, 1, 5, 15])
    vortunit = '10$^{-5}$ s$^{-1}$'
    tendunit = '10$^{-10}$ s$^{-2}$'
    cmap = cm.get_cmap('RdBu_r')

    x = state['longitude'].values
    y = state['latitude'].values
    if 'perturbation_atmosphere_relative_vorticity' in state.keys():
        vortp = state['perturbation_atmosphere_relative_vorticity'].values * 10**5
        vortb = state['base_atmosphere_relative_vorticity'].values * 10**5
        ten_dyn = state['perturbation_atmosphere_relative_vorticity_tendency_from_dynamics'].values * 10**10
        ten_diff = state['perturbation_atmosphere_relative_vorticity_tendency_from_diffusion'].values * 10**10
        u = state['perturbation_eastward_wind'].values + state['base_eastward_wind'].values
        v = state['perturbation_northward_wind'].values + state['base_northward_wind'].values
        if 'perturbation_atmosphere_relative_vorticity_tendency_from_forcing' in state.keys():
            forcing = state['perturbation_atmosphere_relative_vorticity_tendency_from_forcing'].values * 10**10
        else:
            forcing = None
    else:
        vortp = state['atmosphere_relative_vorticity'].values * 10 ** 5
        vortb = np.zeros(vortp.shape)
        ten_dyn = state['atmosphere_relative_vorticity_tendency_from_dynamics'].values * 10 ** 10
        ten_diff = state['atmosphere_relative_vorticity_tendency_from_diffusion'].values * 10 ** 10
        u = state['eastward_wind'].values
        v = state['northward_wind'].values
        if 'atmosphere_relative_vorticity_tendency_from_forcing' in state.keys():
            forcing = state['atmosphere_relative_vorticity_tendency_from_forcing'].values * 10 ** 10
        else:
            forcing = None

    axes = [ax1, ax2, ax3, ax4]
    datas = [vortp+vortb, vortp, ten_dyn, ten_diff]
    titles = ['total vort.', 'pert. vort.', 'dynamics tend.', 'diffusion tend.']
    units = [vortunit, vortunit, tendunit, tendunit]
    levels = [vortlevs, vortlevs, tendlevs, tendlevs]

    for ax, data, title, unit, levs in zip(axes, datas, titles, units, levels):
        cs = ax.contourf(x, y, data, cmap=NonlinearCmap(cmap, levs), levels=levs)
        ax.barbs(x[8::8, 8::8], y[8::8, 8::8], u[8::8, 8::8], v[8::8, 8::8], length=4)
        if forcing is not None:
            ax.contour(x, y, forcing, levels=tendlevs, colors='k')
        ax.set_title('{} [{}]'.format(title, unit), fontsize=8)
        if ax == ax1:
            cb = fig.colorbar(cs, cax=cax1)
            cb.set_ticks(levs)
            cb.set_ticklabels(levs)
        elif ax == ax3:
            cb = fig.colorbar(cs, cax=cax2)
            cb.set_ticks(levs)
            cb.set_ticklabels(levs)
        if ax in [ax2, ax4]:
            ax.set_yticklabels([])
        if ax in [ax1, ax2]:
            ax.set_xticklabels([])
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)

    fig.suptitle('{:%Y-%m-%d %H:%M}'.format(state['time']), x=0.46, fontsize=10)
    savedir = '/Users/nweber/barotropy/testfigs_spectral_debug'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    plt.savefig('{}/baro_{:%Y-%m-%d_%H%M}.png'.format(savedir, state['time']))


def fourpanel_polar(fig, state):
    m = Basemap(projection='npstere', boundinglat=55, lon_0=270, resolution='l')
    x, y = m(state['longitude'].values, state['latitude'].values)

    ax1 = fig.add_axes([0.05, 0.52, 0.40, 0.42])
    ax2 = fig.add_axes([0.47, 0.52, 0.40, 0.42])
    ax3 = fig.add_axes([0.05, 0.05, 0.40, 0.42])
    ax4 = fig.add_axes([0.47, 0.05, 0.40, 0.42])
    cax1 = fig.add_axes([0.89, 0.52, 0.02, 0.42])
    cax2 = fig.add_axes([0.89, 0.05, 0.02, 0.42])

    vortlevs = np.array([-10, -7.5, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 7.5, 10])
    tendlevs = np.array([-15, -5, -1, -0.5, -0.2, -0.05, -0.01, 0.01, 0.05, 0.2, 0.5, 1, 5, 15])
    forclevs = np.array([-20, -15, -10, -5, -0.01, 0.01, 5, 10, 15, 20])
    vortunit = '10$^{-5}$ s$^{-1}$'
    tendunit = '10$^{-10}$ s$^{-2}$'
    cmap = cm.get_cmap('RdBu_r')

    vortp = state['perturbation_atmosphere_relative_vorticity'].values * 10**5
    vortb = state['base_atmosphere_relative_vorticity'].values * 10**5
    ten_dyn = state['perturbation_atmosphere_relative_vorticity_tendency_from_dynamics'].values * 10**10
    ten_diff = state['perturbation_atmosphere_relative_vorticity_tendency_from_diffusion'].values * 10**10
    if 'perturbation_atmosphere_relative_vorticity_tendency_from_forcing' in state.keys():
        forcing = state['perturbation_atmosphere_relative_vorticity_tendency_from_forcing'].values * 10**10
    else:
        forcing = None

    axes = [ax1, ax2, ax3, ax4]
    datas = [vortp+vortb, vortp, ten_dyn, ten_diff]
    titles = ['total vort.', 'pert. vort.', 'dynamics tend.', 'diffusion tend.']
    units = [vortunit, vortunit, tendunit, tendunit]
    levels = [vortlevs, vortlevs, tendlevs, tendlevs]

    for ax, data, title, unit, levs in zip(axes, datas, titles, units, levels):
        m = Basemap(projection='npstere', boundinglat=55, lon_0=270, resolution='l', ax=ax)
        m.drawcoastlines()
        m.drawparallels(np.arange(-85., 86., 5.))
        m.drawmeridians(np.arange(-180., 181., 20.))
        ax.set_title('{} [{}]'.format(title, unit), fontsize=8)
        cs = m.contourf(x, y, data, cmap=NonlinearCmap(cmap, levs), levels=levs)
        if forcing is not None:
            m.contour(x, y, forcing, levels=forclevs, colors='k')
        if ax == ax1:
            cb = fig.colorbar(cs, cax=cax1)
            cb.set_ticks(levs)
            cb.set_ticklabels(levs)
        elif ax == ax3:
            cb = fig.colorbar(cs, cax=cax2)
            cb.set_ticks(levs)
            cb.set_ticklabels(levs)
        if ax in [ax2, ax4]:
            ax.set_yticklabels([])
        if ax in [ax1, ax2]:
            ax.set_xticklabels([])

    fig.suptitle('{:%Y-%m-%d %H:%M}'.format(state['time']), x=0.46, fontsize=10)
    savedir = '/Users/nweber/barotropy/testfigs_spectral'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    plt.savefig('{}/baro_{:%Y-%m-%d_%H%M}.png'.format(savedir, state['time']))

