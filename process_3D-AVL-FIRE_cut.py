# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:14:26 2020

@author: feierabend
"""

import numpy as np
from scipy.interpolate import griddata
import os
import matplotlib.pyplot as plt
import timeit


start_time = timeit.default_timer()

# specify directory and file names
dir_name = r'D:\ZBT\Projekte\Manifold_Modell'
file_name = \
    'Stack_Ratio2_Re3000_p01_IT_1194_Flow_RelativePressure_Pa.npy'

output_dir = 'Case_1'

# specify manifold geometry
n_channels = 16
n_manifolds = 2
flow_dir_y = -1
channel_dy = 5e-4
manifold_dz = 5e-4
manifold_height = (0, 0.24)
manifold_distance = manifold_height[1] - manifold_height[0]
manifold_diameter = 0.01
channel_distance_z = 0.02
channel_0_z = 0.005
z_min = -0.105
z_max = 0.315
x_min = 0.0
x_max = 0.0
y_min = 0.0 - 0.5 * manifold_diameter
y_max = manifold_distance + 0.5 * manifold_diameter

# create output folder
full_output_dir = os.path.join(dir_name, output_dir)
if not os.path.isdir(full_output_dir):
    os.mkdir(full_output_dir)

# load raw data from AVL FIRE 3D cut (previously converted to binary)
raw_data = np.load(os.path.join(dir_name, file_name)).transpose()    
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
data = np.asarray(np.vstack((y,z,p)))
print(timeit.default_timer() - start_time)

print(timeit.default_timer() - start_time)

# create channel centerline coordinates
channel_z = np.asarray([channel_0_z + i * channel_distance_z 
                        for i in range(n_channels)])
ny_channel = int(np.round((y_max - y_min)/channel_dy))
channel_z_positions = np.asarray([channel_0_z + i * channel_distance_z 
                        for i in range(n_channels)])
channel_z = np.zeros((ny_channel, n_channels))
channel_z[:, :] = channel_z_positions
channel_y = np.linspace(manifold_height[0], manifold_height[1], ny_channel)
channel_coords = np.zeros((n_channels, ny_channel, 2))
# set channel y-coordinates
channel_coords[:, :, 0] = channel_y
# set channel z-coordinates
channel_coords[:, :, 1] = channel_z.transpose()
# combine channel coordinates into one array for efficient interpolation
channel_coords_combined = np.asarray([item for item in channel_coords])
channel_coords_combined = \
    channel_coords_combined.reshape(-1, channel_coords_combined.shape[-1])
print(timeit.default_timer() - start_time)

# create manifold centerline coordinates
nz_manifold = int(np.round((z_max - z_min)/manifold_dz)) + 1
manifold_z = np.linspace(z_min, z_max, nz_manifold)
manifold_coords = np.zeros((n_manifolds, nz_manifold, 2))

manifold_coords[0, :, 0] = manifold_height[0]
manifold_coords[1, :, 0] = manifold_height[1]
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
channel_pressure_combined = pressure_combined[:n_channels * ny_channel]
channel_pressure = channel_pressure_combined.reshape(n_channels, ny_channel)
manifold_pressure_combined = pressure_combined[-n_manifolds * nz_manifold:]
manifold_pressure = \
    manifold_pressure_combined.reshape(n_manifolds, nz_manifold)

fig = plt.figure()
plt.plot(channel_coords[5, :, 0], channel_pressure[5, :])
print(timeit.default_timer() - start_time)

# write channel data to files
channel_data = []
for i in range(n_channels):
    channel_data.append(channel_coords[i, :, 0])
    channel_data.append(channel_coords[i, :, 1])
    channel_data.append(channel_pressure[i])
channel_data = np.asarray(channel_data)
np.save(os.path.join(full_output_dir, 'channel_data'), channel_data)

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
for i in range(n_manifolds):
    manifold_data.append(manifold_coords[i, :, 0])
    manifold_data.append(manifold_coords[i, :, 1])
    manifold_data.append(manifold_pressure[i])
manifold_data = np.asarray(manifold_data)
np.save(os.path.join(full_output_dir, 'manifold_data'), manifold_data)
