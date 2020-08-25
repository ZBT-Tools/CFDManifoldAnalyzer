# global imports
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib

# custom library imports
import pemfc

# local imports
from cfd_manifold_analyzer.settings import file_names
from cfd_manifold_analyzer.settings import geometry as geom
from cfd_manifold_analyzer.settings import physical_properties
import cfd_manifold_analyzer.src.cfd_data_processor as cfd_proc
from cfd_manifold_analyzer.settings import model
from cfd_manifold_analyzer.src import streamline
from cfd_manifold_analyzer.src import globals

matplotlib.use('TkAgg')

# specify boundary conditions
reynolds_number = 2300.0
reynolds_number_channel = 200.0
split_ratio = None
manifold_area = geom.manifold_diameter ** 2.0 * np.pi * 0.25
channel_area = geom.channel_diameter ** 2.0 * np.pi * 0.25

# specify directories and file paths
pressure_file_path = \
    os.path.join(file_names.dir_name, file_names.avl_fire_file_3d)
output_dir = os.path.join(file_names.dir_name, file_names.output_dir)

# specify variation parameters
n_cases = 40
case_dict = {
    'number': list(range(1, n_cases + 1)),
    'file_variation_pattern': 'Case_',
    'file_path': pressure_file_path,
    'variation_parameter': 'split_ratio',
    'value_variation': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25,
                        0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                        0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0,
                        0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.16,
                        0.17, 0.18, 0.19]
}

# setup and create the model channels
fluid = pemfc.fluid.dict_factory(model.fluid_dict)
chl = pemfc.channel.Channel(model.channel_dict, fluid)
fluid = pemfc.fluid.dict_factory(model.fluid_dict)
mfd = pemfc.channel.Channel(model.manifold_dict, fluid)

# create variation lists
split_ratios = []
zetas_mfd_mfd = []
zetas_chl_mfd = []
reynolds_mfd = []
reynolds_chl = []

# parameter variation loop
for i in range(n_cases):
    print('processing case', str(i + 1))
    old_file_path = case_dict['file_path']
    case_number = case_dict['number'][i]
    file_path = globals.replace_number(old_file_path,
                                       case_dict['file_variation_pattern'],
                                       case_number)

    # set variation parameter value
    setattr(sys.modules[__name__], case_dict['variation_parameter'],
            case_dict['value_variation'][i])
    split_ratios.append(split_ratio)
    # calculate dependent boundary conditions
    # manifold_velocity = reynolds_number * physical_properties.viscosity \
    #     / (geom.manifold_diameter * physical_properties.density)
    # manifold_mass_flow = \
    #     manifold_velocity * manifold_area * physical_properties.density
    #
    # channel_mass_flow = split_ratio * manifold_mass_flow
    # mass_flow_data = np.array([channel_mass_flow])
    # channel_velocity = \
    #     channel_mass_flow / (physical_properties.density * channel_area)
    # reynolds_chl.append(channel_velocity * geom.channel_diameter *
    #                     physical_properties.density /
    #                     physical_properties.viscosity)
    channel_velocity = reynolds_number_channel * physical_properties.viscosity \
        / (physical_properties.density * geom.channel_diameter)
    channel_mass_flow = channel_velocity * channel_area \
        * physical_properties.density
    manifold_mass_flow = channel_mass_flow / split_ratio
    manifold_velocity = \
        manifold_mass_flow / manifold_area / physical_properties.density
    mass_flow_data = np.array([channel_mass_flow])
    reynolds_mfd.append(manifold_velocity * geom.manifold_diameter
                        * physical_properties.density
                        / physical_properties.viscosity)
    reynolds_chl.append(reynolds_number_channel)
    # load and process 3D AVL FIRE M data
    cfd_data = cfd_proc.CFDTJunctionProcessor(file_path,
                                              mass_flow_data, output_dir)
    cfd_data.process()
    # cfd_data.plot(name_extension='_' + str(case_number))

    # get mass flow data
    mass_flows = cfd_data.mass_flow_data.mass_flows

    # get channel outlet pressure
    x_out_channel = geom.channel_start_vector[0][1] + chl.length
    p_chl_out = cfd_data.channels[0].data_function['pressure'](x_out_channel)
    p_mfd_out = \
        cfd_data.manifolds[0].data_function['pressure'](geom.manifold_range[-1])

    p_mfd_in = \
        cfd_data.manifolds[0].data_function['pressure'](geom.manifold_range[0])
    p_mfd = [p_mfd_in, p_mfd_out]
    p_chl = [None, p_chl_out]

    # channel update
    chl.p_out = p_chl_out
    chl.update(mass_flow_in=channel_mass_flow, update_heat=False)

    # manifold update
    mfd.p_out = p_mfd_out
    mass_source = - channel_mass_flow
    mfd.update(mass_flow_in=manifold_mass_flow,
               mass_source=mass_source, update_heat=False)

    # set length segments according to analysation points
    dx_mfd = [0.0 - geom.manifold_range[0], geom.manifold_range[1] - 0.0]
    dx_chl = geom.channel_length

    # create streamline object from manifold (mfd) to channel (chl)
    sl_mfd_chl = streamline.Streamline()
    sl_mfd_chl.add_point("mfd_in", channel=mfd, idx=0,
                         half_distance=dx_mfd[0], pressure=p_mfd_in)
    sl_mfd_chl.add_point("chl_in", channel=chl, idx=-1,
                         half_distance=dx_chl, pressure=p_chl_out)
    sl_mfd_chl.add_point("mfd_out", channel=mfd, idx=1,
                         half_distance=dx_mfd[1], pressure=p_mfd_out)
    zeta_mfd_mfd = sl_mfd_chl.calculate_zeta(0, 2)
    zetas_mfd_mfd.append(zeta_mfd_mfd)
    p_mfd_mfd = sl_mfd_chl.calculate_pressure_difference(0, 2, zeta_mfd_mfd)
    print('zeta mfd-mfd: ', zeta_mfd_mfd)
    print('calculated pressure difference mfd-mfd: ', p_mfd_mfd)
    print('cfd pressure difference mfd-mfd: ', p_mfd[1] - p_mfd[0])
    zeta_chl_mfd = sl_mfd_chl.calculate_zeta(1, 2)
    zetas_chl_mfd.append(zeta_chl_mfd)
    p_chl_mfd = sl_mfd_chl.calculate_pressure_difference(1, 2, zeta_chl_mfd)
    print('zeta mfd-chl: ', zeta_chl_mfd)
    print('calculated pressure difference mfd-chl: ', p_chl_mfd)
    print('cfd pressure difference mfd-chl: ', p_mfd[1] - p_chl[1])

# sort results
x_array = np.array(split_ratios)
y_array_1 = np.array(zetas_mfd_mfd)
y_array_2 = np.array(zetas_chl_mfd)
y_array_3 = np.array(reynolds_mfd)
y_array_4 = np.array(reynolds_chl)
xy_array = np.vstack((x_array, y_array_1, y_array_2, y_array_3, y_array_4))
xy_array = xy_array[:, xy_array[0].argsort()]

# save results as ascii file
np.savetxt(os.path.join(output_dir, file_names.output_main_name + '.txt'),
           xy_array)

# plot results
import cfd_manifold_analyzer.plot_data
