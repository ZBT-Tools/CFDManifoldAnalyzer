import sys
import os
import pemfc
import numpy as np

# import cfd_manifold_analyzer.src.convert_to_binary
# import cfd_manifold_analyzer.src.process_3d_avl_fire_cut
from cfd_manifold_analyzer.settings import file_names
from cfd_manifold_analyzer.settings import geometry as geom
from cfd_manifold_analyzer.settings import physical_properties
import cfd_manifold_analyzer.src.cfd_data_processor as cfd_proc
from cfd_manifold_analyzer.settings import model

# calculate boundary conditions


reynolds_number = 2300.0
split_ratio = 0.01
manifold_area = geom.manifold_diameter ** 2.0 * np.pi * 0.25
channel_area = geom.channel_diameter ** 2.0 * np.pi * 0.25

manifold_velocity = reynolds_number * physical_properties.viscosity \
    / (geom.manifold_diameter * physical_properties.density)
manifold_mass_flow = \
    manifold_velocity * manifold_area * physical_properties.density

channel_mass_flow = split_ratio * manifold_mass_flow
channel_velocity = \
    manifold_mass_flow / (physical_properties.density * channel_area)

# load and process cfd data
pressure_file_path = \
    os.path.join(file_names.dir_name, file_names.avl_fire_file_3d)
mass_flow_data = np.array([channel_mass_flow])
output_dir = os.path.join(file_names.dir_name, file_names.output_dir)
cfd_data = cfd_proc.CFDTJunctionProcessor(pressure_file_path,
                                          mass_flow_data, output_dir)
cfd_data.process()
cfd_data.plot()

# setup and create the model channels
fluid = pemfc.fluid.dict_factory(model.fluid_dict)
channel = pemfc.channel.Channel(model.channel_dict, fluid)
fluid = pemfc.fluid.dict_factory(model.fluid_dict)
manifold = pemfc.channel.Channel(model.manifold_dict, fluid)

# get mass flow data
mass_flows = cfd_data.mass_flow_data.mass_flows

# get channel outlet pressure
x_out_channel = geom.channel_start_vector[0][1] + channel.length
p_chl_out = cfd_data.channels[0].data_function['pressure'](x_out_channel)
print(p_chl_out)
p_mfd_out = \
    cfd_data.manifolds[0].data_function['pressure'](geom.manifold_range[-1])
print(p_mfd_out)

p_mfd_in = \
    cfd_data.manifolds[0].data_function['pressure'](geom.manifold_range[0])
p_mfd = [p_mfd_in, p_mfd_out]
p_chl = [None, p_chl_out]

# channel update
channel.p_out = p_chl_out
channel.update(mass_flow_in=channel_mass_flow,
               update_heat=False)

# manifold update
manifold.p_out = p_mfd_out
mass_source = - channel_mass_flow
manifold.update(mass_flow_in=manifold_mass_flow,
                mass_source=mass_source, update_heat=False)

print(manifold.velocity)
print(manifold.reynolds)
print(channel.velocity)
mfd = manifold
chl = channel
# set length segments according to analysation points
dx_mfd = [0.0 - geom.manifold_range[0], geom.manifold_range[1] - 0.0]
dx_chl = geom.channel_length


# get tube friction factors
for zeta in manifold.zetas:
    if isinstance(zeta, pemfc.flow_resistance.WallFrictionFlowResistance):
        f_mfd = zeta.value * manifold.d_h / (manifold.dx_node * 0.5)

for zeta in channel.zetas:
    if isinstance(zeta, pemfc.flow_resistance.WallFrictionFlowResistance):
        f_chl = np.average(zeta.value * channel.d_h / (channel.dx_node * 0.5))

# calculate t-junction flow resistance factor within manifold
v1_v0_2 = (mfd.velocity[1] / mfd.velocity[0]) ** 2.0
rho1_rho0 = mfd.fluid.density[1] / mfd.fluid.density[0]

zeta_mfd = 1.0 \
    + (p_mfd[0] - p_mfd[1]) \
    * 2.0 / (mfd.fluid.density[0] * mfd.velocity[0] ** 2.0) \
    - rho1_rho0 * v1_v0_2 \
    - dx_mfd[0] / mfd.d_h * f_mfd[0] \
    - rho1_rho0 * v1_v0_2 * dx_mfd[1] / mfd.d_h * f_mfd[1]

# calculate t-junction flow resistance factor from manifold to channel
v2_v0_2 = (chl.velocity[1] / mfd.velocity[0]) ** 2.0
rho2_rho0 = chl.fluid.density[1] / mfd.fluid.density[0]
zeta_chl = 1.0 \
    + (p_mfd[0] - p_chl[1]) \
    * 2.0 / (mfd.fluid.density[0] * mfd.velocity[0] ** 2.0) \
    - rho2_rho0 * v2_v0_2 \
    - dx_mfd[0] / mfd.d_h * f_mfd[0] \
    - rho2_rho0 * v2_v0_2 * dx_chl / chl.d_h * f_chl

print(zeta_mfd)
print(zeta_chl)

# calculate t-junction flow resistance factor
# including dynamic and tube resistance
zeta_mfd_2 = (p_mfd[1] - p_mfd[0]) \
    * 2.0 / (mfd.fluid.density[0] * mfd.velocity[0] ** 2.0)
zeta_chl_2 = (p_chl[1] - p_mfd[0]) \
    * 2.0 / (mfd.fluid.density[0] * mfd.velocity[0] ** 2.0)

print(zeta_mfd_2)
print(zeta_chl_2)


