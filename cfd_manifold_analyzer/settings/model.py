from . import geometry
from . import physical_properties

# fluid model settings
temperature = 293.15
pressure = 101325.0
nodes = 2

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

# channel model settings
channel_dict = {
    'name': 'Channel',
    'length': geometry.channel_length,
    'cross_sectional_shape': 'circular',
    'diameter': geometry.channel_diameter,
    # 'width': geometry.channel_diameter,
    # 'height': geometry.channel_diameter,
    'p_out': pressure,
    'temp_in': temperature,
    'flow_direction': 1,
    'constant_friction_factor': 0.0,
    'wall_friction': True
}

manifold_dict = {
    'name': 'Manifold',
    'length': geometry.manifold_range[1] - geometry.manifold_range[0],
    'diameter': geometry.manifold_diameter,
    # 'width': geometry.manifold_diameter,
    # 'height': geometry.manifold_diameter,
    'p_out': pressure,
    'temp_in': temperature,
    'flow_direction': 1,
    'constant_friction_factor': 0.0,
    'wall_friction': True
}