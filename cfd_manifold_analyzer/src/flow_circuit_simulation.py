# general imports
import numpy as np
import cProfile
import copy
import sys
import matplotlib.pyplot as plt
import pemfc
# from pemfc import channel as chl
# from pemfc import fluid as fluid
# from pemfc import flow_circuit as flow_circuit
# from pemfc import interpolation as ip

# local module imports
from ..settings import geometry
from ..settings import file_names

n_chl = geometry.n_channels
n_subchl = 1

temperature = 293.15
pressure = 101325.0
nodes = geometry.

channel_dict = {
    'name': 'Channel',
    'length': geometry.channel_length,
    'cross_sectional_shape': 'circular',
    'diameter': geometry.channel_diameter,
    'width': 4e-3,
    'height': 1e-3,
    'p_out': pressure,
    'temp_in': temperature,
    'flow_direction': 1,
    'bend_number': 0,
    'bend_friction_factor': 0.1,
    'constant_friction_factor': 0.2
    }


fluid_dict = {
    'name': 'Cathode Gas',
    'fluid_components': {'O2': 'gas', 'N2': 'gas', 'H2O': 'gas-liquid'},
    'inlet_composition': [0.21, 0.79, 0.0],
    'temp_init': temperature,
    'press_init': pressure,
    'nodes': nodes
}

constant_fluid_dict = {
    'name': 'Air',
    'fluid_components': None,
    'inlet_composition': None,
    'specific_heat': 1007.0,
    'density': 1.20433,
    'viscosity': 1.824e-05,
    'thermal_conductivity': 0.0257,
    'temp_init': temperature,
    'press_init': pressure,
    'nodes': nodes
    }

in_manifold_dict = {
    'name': 'Inlet Manifold',
    'length': 0.27,
    'p_out': channel_dict['p_out'],
    'temp_in': 293.15,
    'flow_direction': 1,
    'width': 12.5e-3,
    'height': 7.5e-3,
    'bend_number': 0,
    'bend_friction_factor': 0.0,
    'constant_friction_factor': -0.5,
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
                                  pemfc.fluid.dict_factory(constant_fluid_dict))
            for i in range(n_chl)]

flow_model = \
    pemfc.flow_circuit.factory2(flow_circuit_dict, in_manifold_dict,
                                out_manifold_dict, channels,
                                channel_multiplier=n_subchl)


x = (pemfc.interpolation.interpolate_1d(flow_model.manifolds[0].x)
     - flow_model.manifolds[0].dx * 0.5) \
    / (flow_model.manifolds[0].length - flow_model.manifolds[0].dx[0])
# x = flow_model.manifolds[0].x / flow_model.manifolds[0].length

flow_model.run(inlet_mass_flow=0.000449642)
q = (flow_model.normalized_flow_distribution - 1.0) * 100.0
reynolds = flow_model.manifolds[0].reynolds[0]
plt.plot(x, q, label='Re={0:.2f}'.format(reynolds), color='k')
plt.show()



# flow_model.update(inlet_mass_flow=0.00059425)
# q = (flow_model.normalized_flow_distribution - 1.0) * 100.0
# reynolds = flow_model.manifolds[0].reynolds[0]
# plt.plot(x, q, label='Re={0:.2f}'.format(reynolds), color='b')
#
# flow_model.update(inlet_mass_flow=0.000297125)
# q = (flow_model.normalized_flow_distribution - 1.0) * 100.0
# reynolds = flow_model.manifolds[0].reynolds[0]
# plt.plot(x, q, label='Re={0:.2f}'.format(reynolds), color='r')

# print('Normalized Flow Distribution: ',
#       flow_model.normalized_flow_distribution)
# np.savetxt('output/flow_distribution.txt',
#            (flow_model.normalized_flow_distribution - 1.0) * 100.0)
m_in = flow_model.manifolds[0]
plt.plot(m_in.p - 101325.0, color='b')
plt.show()
m_out = flow_model.manifolds[1]
plt.plot(m_out.p - 101325.0, color='r')
plt.show()
