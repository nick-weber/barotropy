# -*- coding: utf-8 -*-

"""
Utility functions for use in the other modules.
"""
import numpy as np


def buffer_poles(data):
    nlat = data.shape[0]
    nlon = data.shape[1]

    buffered_data = np.zeros((nlat+2, nlon))
    buffered_data[1:-1, :] = data

    buffered_data[0, :] = np.nanmean(data[0, :])
    buffered_data[-1, :] = np.nanmean(data[-1, :])
    return buffered_data


def unbuffer_poles(buffered_data):
    return buffered_data[1:-1, :]
