import sys
import os
import pemfc
import numpy as np

# import cfd_manifold_analyzer.src.convert_to_binary
# import cfd_manifold_analyzer.src.process_3d_avl_fire_cut
from cfd_manifold_analyzer.settings import file_names
from cfd_manifold_analyzer.settings import geometry
from cfd_manifold_analyzer.settings import physical_properties
import cfd_manifold_analyzer.src.cfd_data_processor as cfd_proc

# load and process cfd data
pressure_file_path = \
    os.path.join(file_names.dir_name, file_names.avl_fire_file_3d)
mass_flow_data = np.array([0.1])
output_dir = os.path.join(file_names.dir_name, file_names.output_dir)
cfd_data = cfd_proc.CFDTJunctionProcessor(pressure_file_path,
                                          mass_flow_data, output_dir)
cfd_data.process()
cfd_data.plot()

# create model fluid
temperature = 293.15
pressure = 101325.0
nodes = 2

# setup and create the model channels

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
# create model channels
n_chl = geometry.n_channels
n_subchl = 1

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
fluid = pemfc.fluid.dict_factory(fluid_dict)
channel = pemfc.channel.Channel(channel_dict, fluid)

manifold_dict = {
    'name': 'Manifold',
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
fluid = pemfc.fluid.dict_factory(fluid_dict)
manifold = pemfc.channel.Channel(manifold_dict, fluid)



# cfd_data.channels[0].plot()
# cfd_data.save()
# channel = pemfc.channel.Channel()
