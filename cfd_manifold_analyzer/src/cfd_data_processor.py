import numpy as np
from scipy.interpolate import griddata
import os
import matplotlib.pyplot as plt
import matplotlib
import timeit
from ..settings import file_names
from ..settings import geometry as geom

matplotlib.use('TkAgg')


class CFDDataChannel:
    def __init__(self, diameter, length, start_vector, direction_vector, dx):
        self.dims = 3
        self.diameter = diameter
        self.length = length
        start_vector = np.asarray(start_vector)
        direction_vector = np.asarray(direction_vector)
        if np.ndim(np.asarray(start_vector)) != self.dims:
            raise ValueError(
                'start_vector must be of size {}'.format(self.dims))
        self.start_vector = start_vector
        if np.ndim(np.asarray(direction_vector)) != self.dims:
            raise ValueError(
                'start_vector must be of size {}'.format(self.dims))
        # normalize direction vector
        try:
            self.direction_vector = \
                direction_vector / np.linalg.norm(direction_vector)
        except FloatingPointError:
            raise FloatingPointError('direction vector must not be zero')
        self.dx = dx
        self.nx = int(np.round(self.length / self.dx)) + 1
        self.pressure = np.zeros(self.nx)
        self.x = np.zeros(self.nx)
        self.coords = np.zeros((self.dims, self.nx))
        self.cord_length = np.linspace(0.0, self.length, self.nx)

    def create_coords(self):
        length_vector = self.length * self.direction_vector
        end_vector = self.start_vector + length_vector
        coords = \
            np.asarray([np.linspace(self.start_vector[i], end_vector[i],
                                    self.nx)
                        for i in range(len(end_vector))])
        self.coords[:] = np.dot(coords, self.direction_vector)
        return coords

    def plot(self, ax=None, xaxis=None, yaxis='pressure'):
        if ax is None:
            fig, ax = plt.figure()
        if xaxis is None:
            xaxis = self.cord_length
        if yaxis == 'pressure':
            yaxis = self.pressure
        else:
            raise ValueError('data not found')
        ax.plot(self.coords[xaxis], yaxis)
        plt.show()


class CFDManifoldProcessor3D:
    def __init__(self, file_path, output_dir):
        self.file_path = file_path
        self.output_dir = output_dir
        # create output folder
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # create channel data objects
        self.channels = []
        for i in range(geom.n_channels):
            start_vector = \
                np.asarray((0.0,
                            geom.manifold_y[0] + 0.5 * geom.manifold_diameter,
                            geom.channel_0_z + i * geom.channel_distance_z))
            direction_vector = np.asarray((0.0, 1.0, 0.0))
            self.channels.append(CFDDataChannel(geom.channel_diameter,
                                                geom.channel_length,
                                                start_vector, direction_vector,
                                                geom.channel_dy))

        # create manifold data objects
        self.manifolds = []
        for i in range(geom.n_manifolds):
            start_vector = \
                np.asarray((0.0, geom.manifold_y[i], geom.bounding_box[-1, 0]))
            direction_vector = np.asarray((0.0, 0.0, 1.0))
            self.manifolds.append(CFDDataChannel(geom.manifold_diameter,
                                                 geom.manifold_length,
                                                 start_vector, direction_vector,
                                                 geom.manifold_dz))
        self.n_channels = len(self.channels)
        self.n_manifolds = len(self.manifolds)
        # self.n_channels = geom.n_channels
        # self.n_manifolds = geom.n_manifolds
        # self.channel_diameter = geom.channel_diameter
        # self.channel_length = geom.channel_length
        # self.channel_distance = geom.channel_distance_z
        # self.channel_dy = geom.channel_dy
        # self.channel_0_z = geom.channel_0_z
        # self.manifold_y = geom.manifold_y
        # self.manifold_dz = geom.manifold_dz
        # self.bounding_box = geom.bounding_box

    def load_data(self):
        if not self.file_path.split('.')[-1] == 'npy':
            raise IOError('file must be numpy binary file with .npy-extension')
        return np.load(self.file_path).transpose()

    def data_to_array(self):
        raw_data = self.load_data()
        # create coordinate and pressure arrays corresponding to coordinate system
        # configuration in AVL FIRE case setup
        x, y, z, p = [], [], [], []
        for i in range(3):
            x.append(raw_data[3 * i])
            y.append(raw_data[3 * i + 1])
            z.append(raw_data[3 * i + 2])
            p.append(raw_data[i + 9])
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        z = np.asarray(z).flatten()
        p = np.asarray(p).flatten()
        coord_array = np.asarray(np.vstack((x, y, z))).transpose()
        value_array = p
        combined_array = np.asarray(np.vstack((x, y, z, p)))
        return combined_array, coord_array, value_array

    def create_channel_coords(self):
        return np.asarray([channel.create_coords().transpose()
                           for channel in self.channels])

    def create_manifold_coords(self):
        return np.asarray([manifold.create_coords().transpose()
                           for manifold in self.manifolds])

    def interpolate_data(self):
        channel_coords = self.create_channel_coords()
        # combine channel coordinates into one array for efficient interpolation
        channel_coords_combined = np.asarray([item for item in channel_coords])
        channel_coords_combined = \
            channel_coords_combined.reshape(-1,
                                            channel_coords_combined.shape[-1])
        manifold_coords = self.create_manifold_coords()
        # combine manifold coordinates into one array for efficient interpolation
        manifold_coords_combined = \
            np.asarray([item for item in manifold_coords])
        manifold_coords_combined = \
            manifold_coords_combined.reshape(-1,
                                             manifold_coords_combined.shape[-1])

        # combine all coordinates into one array for efficient interpolation
        coords_combined = np.concatenate((channel_coords_combined,
                                          manifold_coords_combined), axis=0)
        # get data coordinates and values from raw data file
        combined_array, data_points, data_values = self.data_to_array()
        # interpolate centerline (manifold and channels) pressure values
        # channel_pressure = np.zeros((n_channels, ny_channel))
        # for i in range(n_channels):
        pressure_combined = griddata(data_points, data_values, coords_combined)
        # split into separate channel and manifold pressure arrays
        nx_channel = self.channels[0].nx
        channel_pressure_combined = \
            pressure_combined[:self.n_channels * nx_channel]
        channel_pressure = \
            channel_pressure_combined.reshape(self.n_channels, nx_channel)
        nx_manifold = self.manifolds[0].nx
        manifold_pressure_combined = \
            pressure_combined[-self.n_manifolds * nx_manifold:]
        manifold_pressure = \
            manifold_pressure_combined.reshape(self.n_manifolds, nx_manifold)
        # assign values to channel data
        for i, channel in enumerate(self.channels):
            channel.pressure[:] = channel_pressure[i]
        for i, manifold in enumerate(self.manifolds):
            manifold.pressure[:] = manifold_pressure[i]
