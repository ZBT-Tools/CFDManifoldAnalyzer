import numpy as np


class Point:
    def __init__(self, name, diameter=None, density=None,
                 viscosity=None, velocity=None, pressure=None):
        self.diameter = diameter
        self.density = density
        self.viscosity = viscosity
        self.velocity = velocity
        self.pressure = pressure

    def get_dynamic_pressure(self):
        return 0.5 * self.density * self.velocity ** 2.0

    def get_static_pressure(self):
        return self.pressure

    def get_total_pressure(self):
        return self.pressure + self.get_dynamic_pressure()


class Streamline:
    def __init__(self):
        self.points = []

    def add_point(self, name, diameter=None, density=None,
                  viscosity=None, velocity=None, pressure=None):
        self.points.append(Point(name, diameter, density, viscosity, velocity,
                                 pressure))
