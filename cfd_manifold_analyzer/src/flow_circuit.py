# general imports
import os
import sys
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pemfc

from ..settings import geometry
from ..settings import file_names
from . import cfd_data_processor


class DataFlowCircuit(pemfc.flow_circuit.ParallelFlowCircuit):
    def __init__(self, dict_flow_circuit, manifolds, channels,
                 n_subchannels=1.0, **kwargs):

        super().__init__(dict_flow_circuit, manifolds, channels,
                         n_subchannels, **kwargs)

    def single_loop(self, inlet_mass_flow=None, update_channels=True):
        pass

    def update(self, inlet_mass_flow=None, calc_distribution=None):
        pass

    def process_data(self, cfd_data, data_name='pressure'):
        if not isinstance(cfd_data, cfd_data_processor.CFDManifoldProcessor):
            raise TypeError('cfd_data must be CFDManifoldProcessor object')
        if not cfd_data.is_processed:
            cfd_data.process()
        for i, channel in enumerate(self.channels):
            data_channel = cfd_data.channels[i]
            x_data = np.dot(data_channel.coords, data_channel.direction_vector)
            x = np.linspace(x_data[0], x_data[-1], channel.n_nodes)
            cfd_data.channels[0].direction_vector *
            channel.pressure[:] = \
                cfd_data.channels[i].data_function[data_name](channel.x)


def factory(dict_circuit, dict_in_manifold, dict_out_manifold,
            channels, channel_multiplier=1.0):
    if not isinstance(channels, (list, tuple)):
        raise TypeError('argument channels must be a list with objects of type '
                        'Channel')
    if not isinstance(channels[0], pemfc.channel.Channel):
        raise TypeError('argument channels must be a list with objects of type '
                        'Channel')

    n_channels = len(channels)
    in_manifold_fluid = channels[0].fluid.copy()
    in_manifold_fluid.rescale(n_channels + 1)
    out_manifold_fluid = in_manifold_fluid.copy()

    manifolds = [pemfc.channel.Channel(dict_in_manifold, in_manifold_fluid),
                 pemfc.channel.Channel(dict_out_manifold, out_manifold_fluid)]

    return DataFlowCircuit(dict_circuit, manifolds, channels,
                           n_subchannels=channel_multiplier)
