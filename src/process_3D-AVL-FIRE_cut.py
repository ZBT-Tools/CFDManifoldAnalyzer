import numpy as np
from scipy.interpolate import griddata
import os
import matplotlib.pyplot as plt
import timeit
from settings.file_names import *
from settings.geometry import *


start_time = timeit.default_timer()

# create output folder
full_output_dir = os.path.join(dir_name, output_dir)
if not os.path.isdir(full_output_dir):
    os.makedirs(full_output_dir)

# load raw data from AVL FIRE 3D cut (previously converted to binary)
file_name = avl_fire_file_3d.split('.')[0] + '.npy'
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
data = np.asarray(np.vstack((y, z, p)))
print(timeit.default_timer() - start_time)

print(timeit.default_timer() - start_time)

# create channel centerline coordinates
ny_channel = \
    int(np.round((manifold_height_y[1] - manifold_height_y[0]) / channel_dy))
channel_z_positions = np.asarray([channel_0_z + i * channel_distance_z
                                  for i in range(n_channels)])
channel_z = np.zeros((ny_channel, n_channels))
channel_z[:, :] = channel_z_positions
channel_y = np.linspace(manifold_height_y[0], manifold_height_y[1], ny_channel)
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
nz_manifold = int(np.round((z_ext[1] - z_ext[0])/manifold_dz)) + 1
manifold_z = np.linspace(z_ext[0], z_ext[1], nz_manifold)
manifold_coords = np.zeros((n_manifolds, nz_manifold, 2))

manifold_coords[0, :, 0] = manifold_height_y[0]
manifold_coords[1, :, 0] = manifold_height_y[1]
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
np.save(os.path.join(full_output_dir, channel_data_file), channel_data)

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
np.save(os.path.join(full_output_dir, manifold_data_file), manifold_data)
