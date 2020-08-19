import numpy as np
import pemfc


class Point:
    def __init__(self, name, diameter=None, half_distance=None, density=None,
                 viscosity=None, velocity=None, pressure=None,
                 friction_factor=None):
        self.name = name
        self.half_distance = half_distance
        self.diameter = diameter
        self.density = density
        self.viscosity = viscosity
        self.velocity = velocity
        self.pressure = pressure
        self.friction_factor = friction_factor
        self.number = None

    def get_dynamic_pressure(self):
        return 0.5 * self.density * self.velocity ** 2.0

    def get_static_pressure(self):
        return self.pressure

    def get_total_pressure(self):
        return self.pressure + self.get_dynamic_pressure()


class Streamline:
    def __init__(self):
        self.points = []

    def add_point(self, name, channel=None, idx=None,
                  diameter=None, half_distance=None, density=None,
                  viscosity=None, velocity=None, pressure=None,
                  friction_factor=None):

        if isinstance(channel, pemfc.channel.Channel):
            if diameter is None:
                diameter = channel.d_h
            if half_distance is None:
                half_distance = channel.dx_node * 0.5

            if idx is None:
                if density is None:
                    density = np.average(channel.fluid.density)
                if viscosity is None:
                    viscosity = np.average(channel.fluid.viscosity)
                if velocity is None:
                    velocity = np.average(channel.velocity)
                if pressure is None:
                    pressure = np.average(channel.pressure)
                if friction_factor is None:
                    friction_factor = 0.0
                    for zeta in channel.zetas:
                        if isinstance(zeta,
                                      pemfc.flow_resistance.
                                      WallFrictionFlowResistance):
                            friction_factor += \
                                np.average(zeta.value * channel.d_h
                                           / (channel.dx_node * 0.5))
            else:
                if density is None:
                    density = channel.fluid.density[idx]
                if viscosity is None:
                    viscosity = channel.fluid.viscosity[idx]
                if velocity is None:
                    velocity = channel.velocity[idx]
                if pressure is None:
                    pressure = channel.pressure[idx]
                if friction_factor is None:
                    friction_factor = 0.0
                    for zeta in channel.zetas:
                        if isinstance(zeta,
                                      pemfc.flow_resistance.
                                      WallFrictionFlowResistance):
                            friction_factor += zeta.value[idx] * channel.d_h \
                                / channel.dx_node[idx]

        self.points.append(Point(name, diameter, half_distance, density,
                                 viscosity, velocity, pressure,
                                 friction_factor))
        self.points[-1].number = len(self.points) - 1

    def calculate_zeta(self, point_a, point_b):
        a = self.points[point_a]
        b = self.points[point_b]
        dyn_pressure_ratio = \
            b.density / a.density * (b.velocity / a.velocity) ** 2.0
        return 1.0 + (a.pressure - b.pressure) \
            * 2.0 / (a.density * a.velocity ** 2.0) \
            - dyn_pressure_ratio \
            * (1.0 + b.half_distance / b.diameter * b.friction_factor) \
            - a.half_distance / a.diameter * a.friction_factor

    def calculate_pressure_difference(self, point_a, point_b, zeta):
        a = self.points[point_a]
        b = self.points[point_b]
        a_dyn_pressure = 0.5 * a.density * a.velocity ** 2.0
        b_dyn_pressure = 0.5 * b.density * b.velocity ** 2.0
        return a_dyn_pressure - b_dyn_pressure \
            - a_dyn_pressure \
            * (a.half_distance / a.diameter * a.friction_factor) \
            - b_dyn_pressure \
            * (b.half_distance / b.diameter * b.friction_factor) \
            - a_dyn_pressure * zeta
