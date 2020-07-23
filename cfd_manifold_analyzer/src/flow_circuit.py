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


class DataChannel(pemfc.channel.IncompressibleFluidChannel):
    def __init__(self, channel_dict, fluid, number=None):
        super().__init__(channel_dict, fluid, number)

    def update_flow(self, update_fluid=False):
        self.calc_flow_velocity()
        if update_fluid:
            self.fluid.update(self.temperature, self.pressure)

    def update_heat(self, wall_temp=None, heat_flux=None, update_fluid=True,
                    enthalpy_source=None, channel_factor=1.0):
        pass


class DataFlowCircuit(pemfc.flow_circuit.ParallelFlowCircuit):
    def __init__(self, dict_flow_circuit, manifolds, channels,
                 n_subchannels=1.0, **kwargs):

        super().__init__(dict_flow_circuit, manifolds, channels,
                         n_subchannels, **kwargs)
        self.initialize = False
        self.calc_distribution = False

    def single_loop(self, inlet_mass_flow=None, update_channels=True):
        raise NotImplementedError('use process function to update data')

    def update(self, inlet_mass_flow=None, channel_mass_flow=None,
               calc_distribution=False):
        if inlet_mass_flow is None or channel_mass_flow is None:
            raise ValueError('argument inlet_mass_flow and channel_mass_flow '
                             'must not be None')
        self.channel_mass_flow[:] = channel_mass_flow
        self.mass_flow_in = inlet_mass_flow
        super().update(inlet_mass_flow=inlet_mass_flow)

    def set_manifold_pressure(self, cfd_data):
        cfd_manifolds = cfd_data.manifolds
        for i, manifold in self.manifolds:


    def set_linear_channel_pressure(self, cfd_data):
        cfd_channels = cfd_data.channels
        for i, channel in self.channels:
            # interpolate pressure inlet and outlet values for channels
            # from linear segments
            edge_nodes = \
                np.array([[cfd_channels[i].x[0]], [cfd_channels[i].x[-1]]])
            p_edges = cfd_channels[i].linear_values(edge_nodes)
            # set linear pressure distribution in channel objects
            channel.pressure[:] = \
                np.linspace(p_edges[0, 0], p_edges[1, 0], channel.n_nodes)

    def set_pressure(self, cfd_data):
        self.set_linear_channel_pressure(cfd_data)
        self.set_manifold_pressure(cfd_data)

    def process(self, cfd_data):
        if not isinstance(cfd_data, cfd_data_processor.CFDManifoldProcessor):
            raise TypeError('cfd_data must be object of '
                            'type CFDManifoldProcessor')
        if not cfd_data.is_processed:
            cfd_data.process()

        channel_mass_flow = cfd_data.mass_flow_data.mass_flows
        inlet_mass_flow = cfd_data.mass_flow_data.total_mass_flow
        self.update(inlet_mass_flow=inlet_mass_flow,
                    channel_mass_flow=channel_mass_flow)

        self.process_channels(cfd_data)


def factory(dict_circuit, dict_in_manifold, dict_out_manifold,
            channels, channel_multiplier=1.0):
    if not isinstance(channels, (list, tuple)):
        raise TypeError('argument channels must be a list with objects of type '
                        'Channel')
    if not isinstance(channels[0], DataChannel):
        raise TypeError('argument channels must be a list with objects of type '
                        'Channel')

    n_channels = len(channels)
    in_manifold_fluid = channels[0].fluid.copy()
    in_manifold_fluid.rescale(n_channels + 1)
    out_manifold_fluid = in_manifold_fluid.copy()

    manifolds = [DataChannel(dict_in_manifold, in_manifold_fluid),
                 DataChannel(dict_out_manifold, out_manifold_fluid)]

    return DataFlowCircuit(dict_circuit, manifolds, channels,
                           n_subchannels=channel_multiplier)
