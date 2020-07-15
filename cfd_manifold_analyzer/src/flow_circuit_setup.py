# general imports
import numpy as np
import cProfile
import copy
import sys
import matplotlib.pyplot as plt
import pemfc


# local module imports
from ..settings import geometry
from ..settings import file_names
from ..settings import physical_properties
from . import flow_circuit

n_chl = geometry.n_channels
n_subchl = 1

temperature = 293.15
pressure = 101325.0
nodes = 100

channel_dict = {
    'name': 'Channel',
    'length': geometry.channel_length,
    'cross_sectional_shape': 'circular',
    'diameter': geometry.channel_diameter,
    'width': geometry.channel_diameter,
    'height': geometry.channel_diameter,
    'p_out': pressure,
    'temp_in': temperature,
    'flow_direction': 1,
    'bend_number': 0,
    'bend_friction_factor': 0.1,
    'constant_friction_factor': 0.2
    }

fluid_dict = {
    'name': 'Air',
    'specific_heat': physical_properties.specific_heat,
    'density': physical_properties.density,
    'viscosity': physical_properties.viscosity,
    'thermal_conductivity': physical_properties.thermal_conductivity,
    'temp_init': temperature,
    'press_init': pressure,
    'nodes': nodes
    }

in_manifold_dict = {
    'name': 'Inlet Manifold',
    'length': geometry.manifold_length,
    'p_out': pressure,
    'temp_in': temperature,
    'flow_direction': 1,
    'width': geometry.manifold_diameter,
    'height': geometry.manifold_diameter,
    'bend_number': 0,
    'bend_friction_factor': 0.0,
    'constant_friction_factor': -0.1,
    'flow_split_factor': 0.0,
    'wall_friction': False
}

out_manifold_dict = copy.deepcopy(in_manifold_dict)
out_manifold_dict['name'] = 'Outlet Manifold'
out_manifold_dict['constant_friction_factor'] = 0.0
out_manifold_dict['flow_split_factor'] = 0.0

flow_circuit_dict = {
    'name': 'Flow Circuit',
    'type': 'ModifiedKoh',
    'shape': 'U'
    }

channels = [pemfc.channel.Channel(channel_dict,
                                  pemfc.fluid.dict_factory(fluid_dict))
            for i in range(n_chl)]

flow_model = \
    flow_circuit.factory(flow_circuit_dict, in_manifold_dict,
                         out_manifold_dict, channels,
                         channel_multiplier=n_subchl)
