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
        self.diameter = diameter
        self.length = length
        start_vector = np.asarray(start_vector)
        direction_vector = np.asarray(direction_vector)
        if np.ndim(np.asarray(start_vector)) != 3:
            raise ValueError('start_vector must have dimension of 3')
        self.start_vector = start_vector
        if np.ndim(np.asarray(direction_vector)) != 3:
            raise ValueError('direction_vector must have dimension of 3')
        # normalize direction vector
        try:
            self.direction_vector = \
                direction_vector / np.linalg.norm(direction_vector)
        except FloatingPointError:
            raise FloatingPointError('direction vector must not be zero')
        self.dx = dx
        self.nx = int(np.round(self.length / self.dx)) + 1
        self.pressure = np.zeros(self.nx)

    def create_coords(self):
        length_vector = self.length * self.direction_vector
        end_vector = self.start_vector + length_vector
        coords = \
            np.asarray([np.linspace(self.start_vector[i], end_vector[i],
                                    self.nx)
                        for i in range(len(end_vector))])
        return coords


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
        if not file_name.split('.')[-1] == 'npy':
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

    #     # create channel centerline coordinates
    #     ny_channel = \
    #         int(np.round(self.channel_length / self.channel_dy))
    #     channel_z_positions = \
    #         np.asarray([self.channel_0_z + i * self.channel_distance
    #                     for i in range(self.n_channels)])
    #     channel_z = np.zeros((ny_channel, self.n_channels))
    #     channel_z[:, :] = channel_z_positions
    #     channel_y = \
    #         np.linspace(self.manifold_y[0] + geom.manifold_diameter * 0.5,
    #                     self.manifold_y[1] - geom.manifold_diameter * 0.5,
    #                     ny_channel)
    #     channel_coords = np.zeros((geom.n_channels, ny_channel, 3))
    #     # set channel y-coordinates
    #     channel_coords[:, :, 1] = channel_y
    #     # set channel z-coordinates
    #     channel_coords[:, :, 2] = channel_z.transpose()
    #     return channel_coords
    #
    # def create_manifold_coords(self):
    #     # create manifold centerline coordinates
    #     nz_manifold = \
    #         int(np.round(
    #             (self.bounding_box[-1, 1] - self.bounding_box[-1, 0]) /
    #              self.manifold_dz)) + 1
    #     manifold_z = np.linspace(geom.z_ext[0], geom.z_ext[1], nz_manifold)
    #     manifold_coords = np.zeros((geom.n_manifolds, nz_manifold, 2))
    #
    #     manifold_coords[0, :, 0] = geom.manifold_y[0]
    #     manifold_coords[1, :, 0] = geom.manifold_y[1]
    #     manifold_coords[0, :, 1] = manifold_z
    #     manifold_coords[1, :, 1] = manifold_z

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
        channel_pressure_combined = \
            pressure_combined[:self.n_channels * ny_channel]
        channel_pressure = \
            channel_pressure_combined.reshape(self.n_channels, ny_channel)
        manifold_pressure_combined = \
            pressure_combined[-self.n_manifolds * nz_manifold:]
        manifold_pressure = \
            manifold_pressure_combined.reshape(self.n_manifolds, nz_manifold)
        # assign values to channel data
        for i, channel in enumerate(self.channels):
            channel.pressure[:] = channel_pressure[i]
        for i, manifold in enumerate(self.manifolds):
            manifold.pressure[:] = manifold_pressure[i]


start_time = timeit.default_timer()
# create output folder
full_output_dir = os.path.join(file_names.dir_name, file_names.output_dir)
if not os.path.isdir(full_output_dir):
    os.makedirs(full_output_dir)

# load raw data from AVL FIRE 3D cut (previously converted to binary)
file_name = file_names.avl_fire_file_3d.split('.')[0] + '.npy'
raw_data = np.load(os.path.join(file_names.dir_name, file_name)).transpose()
print(timeit.default_timer() - start_time)

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
data_points = np.asarray(np.vstack((y, z))).transpose()
data_values = p
data = np.asarray(np.vstack((y, z, p)))
print(timeit.default_timer() - start_time)

print(timeit.default_timer() - start_time)

# create channel centerline coordinates
ny_channel = \
    int(np.round(geom.channel_length / geom.channel_dy))
channel_z_positions = \
    np.asarray([geom.channel_0_z + i * geom.channel_distance_z
                for i in range(geom.n_channels)])
channel_z = np.zeros((ny_channel, geom.n_channels))
channel_z[:, :] = channel_z_positions
channel_y = \
    np.linspace(geom.manifold_y[0] + geom.manifold_diameter * 0.5,
                geom.manifold_y[1] - geom.manifold_diameter * 0.5,
                ny_channel)
channel_coords = np.zeros((geom.n_channels, ny_channel, 2))
# set channel y-coordinates
channel_coords[:, :, 0] = channel_y
# set channel z-coordinates
channel_coords[:, :, 1] = channel_z.transpose()
print('channels_coords')
print(channel_coords.shape)
# combine channel coordinates into one array for efficient interpolation
# channel_coords_combined = np.asarray([item for item in channel_coords])
#print(channel_coords_combined.shape)
channel_coords_combined = \
    channel_coords.reshape(-1, channel_coords.shape[-1])
print(channel_coords_combined.shape)
print(timeit.default_timer() - start_time)

# create manifold centerline coordinates
nz_manifold = \
    int(np.round((geom.z_ext[1] - geom.z_ext[0])/geom.manifold_dz)) + 1
manifold_z = np.linspace(geom.z_ext[0], geom.z_ext[1], nz_manifold)
manifold_coords = np.zeros((geom.n_manifolds, nz_manifold, 2))

manifold_coords[0, :, 0] = geom.manifold_y[0]
manifold_coords[1, :, 0] = geom.manifold_y[1]
manifold_coords[0, :, 1] = manifold_z
manifold_coords[1, :, 1] = manifold_z
# combine manifold coordinates into one array for efficient interpolation
manifold_coords_combined = np.asarray([item for item in manifold_coords])
manifold_coords_combined = \
    manifold_coords_combined.reshape(-1, manifold_coords_combined.shape[-1])

# combine all coordinates into one array for efficient interpolation
coords_combined = \
    np.concatenate((channel_coords_combined, manifold_coords_combined), axis=0)

print(coords_combined.shape)
# interpolate centerline (manifold and channels) pressure values
# channel_pressure = np.zeros((n_channels, ny_channel))
# for i in range(n_channels):
pressure_combined = \
    griddata(data_points, data_values, coords_combined)
    
# split into separate channel and manifold pressure arrays
channel_pressure_combined = pressure_combined[:geom.n_channels * ny_channel]
channel_pressure = \
    channel_pressure_combined.reshape(geom.n_channels, ny_channel)
manifold_pressure_combined = pressure_combined[-geom.n_manifolds * nz_manifold:]
manifold_pressure = \
    manifold_pressure_combined.reshape(geom.n_manifolds, nz_manifold)

fig = plt.figure()
plt.plot(channel_coords[5, :, 0], channel_pressure[5, :])
plt.show()
print(timeit.default_timer() - start_time)

# write channel data to files
channel_data = []
for i in range(geom.n_channels):
    channel_data.append(channel_coords[i, :, 0])
    channel_data.append(channel_coords[i, :, 1])
    channel_data.append(channel_pressure[i])
channel_data = np.asarray(channel_data)
np.save(os.path.join(full_output_dir, file_names.channel_data_file),
        channel_data)

print(timeit.default_timer() - start_time)

# # interpolate manifold centerline pressure
# manifold_pressure = np.zeros((2, nz_manifold))
# for i in range(manifold_pressure.shape[0]):
#     manifold_pressure[i] = \
#         griddata(data_points, data_values, manifold_coords[i])
# print(timeit.default_timer() - start_time)

fig = plt.figure()
plt.plot(manifold_coords[0, :, 1], manifold_pressure[0])
# print(timeit.default_timer() - start_time)

# write manifold data to file
manifold_data = []
for i in range(geom.n_manifolds):
    manifold_data.append(manifold_coords[i, :, 0])
    manifold_data.append(manifold_coords[i, :, 1])
    manifold_data.append(manifold_pressure[i])
manifold_data = np.asarray(manifold_data)
np.save(os.path.join(full_output_dir, file_names.manifold_data_file),
        manifold_data)
