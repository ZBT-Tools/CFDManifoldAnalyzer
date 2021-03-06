# general imports
import os
import numpy as np
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import math

# local module imports
from cfd_manifold_analyzer.settings import file_names
from cfd_manifold_analyzer.settings import geometry as geom
from cfd_manifold_analyzer.settings import physical_properties as props


def add_source(var, source, direction=1, tri_mtx=None):
    """
    Add discrete 1d source of length n-1 to var of length n
    :param var: 1d array of quantity variable
    :param source: 1d array of source to add to var
    :param direction: flow direction (1: along array counter, -1: opposite to
    array counter)
    :param tri_mtx: if triangle matrix (2D array, nxn) is not provided,
    it will be created temporarily
    :return:
    """
    n = len(var) - 1
    if len(source) != n:
        raise ValueError('parameter source must be of length (var-1)')
    if direction == 1:
        if tri_mtx is None:
            ones = np.zeros((n, n))
            ones.fill(1.0)
            fwd_mat = np.tril(ones)
        else:
            fwd_mat = tri_mtx
        var[1:] += np.matmul(fwd_mat, source)
    elif direction == -1:
        if tri_mtx is None:
            ones = np.zeros((n, n))
            ones.fill(1.0)
            bwd_mat = np.triu(ones)
        else:
            bwd_mat = tri_mtx
        var[:-1] += np.matmul(bwd_mat, source)
    else:
        raise ValueError('parameter direction must be either 1 or -1')
    return var


def calc_pressure_drop(velocity, density, zeta, flow_direction=1):
    """
    Calculates the element-wise pressure drop in the channel
    """
    # if np.shape(velocity)[0] != (np.shape(density)[0] + 1):
    #     raise ValueError('velocity array must be provided as a 1D'
    #                      'nodal array (n+1), while the other settings arrays '
    #                      'must be element-wise (n)')
    v1 = velocity[:-1]
    v2 = velocity[1:]
    a = v1 ** 2.0 * zeta
    b = (v2 ** 2.0 - v1 ** 2.0) * flow_direction
    return (a + b) * density * 0.5


def linear_coefficients(x, y, method='2-points'):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError('x and y must have equal shapes')
    if np.ndim(x) != 1 or x.shape[-1] < 2:
        raise ValueError('x and y must be one-dimensional array with at '
                         'least two entries')
    if method == '2-points' or x.shape[-1] == 2:
        m = (y[1] - y[0]) / (x[1] - x[0])
        b = y[0] - m * x[0]
        return np.asarray((b, m))
    elif method == 'polyfit':
        return np.polynomial.polynomial.polyfit(x, y, 1)
    else:
        raise NotImplementedError


def find_linear_segment_interceptions(coeffs):
    coeffs = np.asarray(coeffs)
    if not coeffs.shape[0] == 2 and not coeffs.shape[0] == 2:
        raise ValueError('parameter coeffs must be numpy array with the first '
                         'two dimensions of shape (2,2)')
    m = coeffs[1]
    b = coeffs[0]
    x = (b[1] - b[0])/(m[0] - m[1])
    y = m[0] * x + b[0]
    return np.array((x, y))


def polyval(x, coeffs):
    return np.polynomial.polynomial.polyval(x, coeffs)


# specify directory and file names
output_dir_name = os.path.join(file_names.dir_name, file_names.output_dir)
manifold_file_name = os.path.join(output_dir_name,
                                  file_names.manifold_data_file + '.npy')
channel_file_name = os.path.join(output_dir_name,
                                 file_names.channel_data_file + '.npy')
mass_flow_file_name = os.path.join(output_dir_name,
                                   file_names.mass_flow_data_file + '.npy')

print(os.path.abspath(output_dir_name))
# load manifold data and assign coordinates and pressure values
# to individual arrays; coordinates correspond to AVL FIRE case setup
manifold_data = np.load(manifold_file_name)
y_manifold = np.asarray([manifold_data[3 * i]
                         for i in range(geom.n_manifolds)])
z_manifold = np.asarray([manifold_data[3 * i + 1]
                         for i in range(geom.n_manifolds)])
p_manifold = np.asarray([manifold_data[3 * i + 2]
                         for i in range(geom.n_manifolds)])

# calculate manifold junction coordinates
junction_width = geom.channel_diameter
z_junction = np.asarray([geom.channel_0_z + geom.channel_distance_z * i
                         for i in range(geom.n_channels)])
z_junction_in = z_junction - junction_width * 0.5
z_junction_out = z_junction + junction_width * 0.5

# make square function for better display of junction coordinates
t = 2.0 * np.pi * 1.0 / geom.channel_distance_z \
    * (z_manifold[0] - z_junction_in[0])
junction_square = 0.5 * signal.square(t, duty=0.25) + 0.5

# interpolate pressure at manifold junctions
p_manifold_function = []
p_manifold_junction = np.zeros((geom.n_manifolds, z_junction.shape[0]))
p_junction_in = np.zeros((geom.n_manifolds, z_junction.shape[0]))
p_junction_out = np.zeros((geom.n_manifolds, z_junction.shape[0]))
dp_junction = np.zeros((geom.n_manifolds, z_junction.shape[0]))
for i in range(geom.n_manifolds):
    p_manifold_function.append(
        interpolate.interp1d(z_manifold[i], p_manifold[i], kind='cubic'))
    p_manifold_junction[i] = p_manifold_function[i](z_junction)
    p_junction_in[i] = p_manifold_function[i](z_junction_in)
    p_junction_out[i] = p_manifold_function[i](z_junction_out)
    dp_junction[i] = p_junction_out[i] - p_junction_in[i]

channel_mass_flows = np.load(mass_flow_file_name)
total_mass_flow = channel_mass_flows.sum()
manifold_mass_flows = np.zeros((geom.n_manifolds,
                                channel_mass_flows.shape[0] + 1))
manifold_mass_flows[:, :] = total_mass_flow
for i in range(geom.n_manifolds):
    add_source(manifold_mass_flows[i], -channel_mass_flows, 
               direction=1)

manifold_area = geom.manifold_diameter ** 2.0 * 0.25 * math.pi
manifold_volume_flow = manifold_mass_flows / props.density
manifold_velocity = manifold_volume_flow / manifold_area
v1 = manifold_velocity[:, :-1]
v2 = manifold_velocity[:, 1:]
zeta_junction = np.zeros(v1.shape)
for i in range(geom.n_manifolds):
    zeta_junction[i] = \
        (2.0 * dp_junction[i] / props.density - (v2[i] ** 2.0 - v1[i] ** 2.0)
         * geom.manifold_flow_direction[i][-1]) / (v1[i] ** 2.0)

# test
n_res = 10000
z_manifold_min = []
z_manifold_max = []
p_manifold_min = []
p_manifold_max = []
dp_junction_2 = []
dp_junction_3 = []
zeta_junction_2 = []
zeta_junction_3 = []
zeta_junction_idelchik = []
zeta_junction_fit = []
velocity_ratio = []
lin_seg_interceptions = []
for i in range(geom.n_manifolds):
    z_manifold_res = np.linspace(z_junction_in[0] - 0.05,
                                 z_junction_out[-1] + 0.03, n_res)
    # z_manifold_res = np.linspace(z_min, z_max, n_res)
    p_manifold_res = p_manifold_function[i](z_manifold_res)
    z_test = z_manifold_res # [400:600]
    p_test = p_manifold_res # [400:600]
    fig, ax1 = plt.subplots()
    wl = np.int(np.round(n_res/10) // 2 * 2 + 1)
    po = 5
    ax1.plot(z_test, p_test)
    p_test = signal.savgol_filter(p_test, window_length=wl, polyorder=po)
    ax1.grid(True, axis='x')
    ax1.set_xticks(np.arange(-0.02, z_test[-1], 0.01))
    grad_p_test = np.gradient(p_test)
    grad_p_test = signal.savgol_filter(grad_p_test,
                                       window_length=wl, polyorder=po)
    # grad_p_test = np.gradient(grad_p_test)
    # grad_p_test = signal.savgol_filter(grad_p_test,
    #                                     window_length=wl, polyorder=po)
    ax2 = ax1.twinx()
    ax2.plot(z_test, grad_p_test)
    # plt.show()

    id_manifold_min = signal.argrelmin(grad_p_test, order=3)[0]
    id_manifold_max = signal.argrelmax(grad_p_test, order=3)[0]
    print(len(id_manifold_min))
    print(len(id_manifold_max))
    p_manifold_min.append(p_manifold_res[id_manifold_max])
    p_manifold_max.append(p_manifold_res[id_manifold_min])
    z_manifold_min.append(z_manifold_res[id_manifold_max])
    z_manifold_max.append(z_manifold_res[id_manifold_min])
    dp_junction_2.append(p_manifold_max[i] - p_manifold_min[i])
    zeta_junction_2.append((2.0 * dp_junction_2[i] / props.density
                            - (v2[i] ** 2.0 - v1[i] ** 2.0)
                            * geom.manifold_flow_direction[i][-1])
                           / (v1[i] ** 2.0))
    min_tangent_coeff = []
    max_tangent_coeff = []
    width = 1
    for j in range(len(id_manifold_min)):
        z = (z_manifold_res[id_manifold_min[j] - width],
             z_manifold_res[id_manifold_min[j] + width])
        p = (p_manifold_res[id_manifold_min[j] - width],
             p_manifold_res[id_manifold_min[j] + width])
        min_tangent_coeff.append(linear_coefficients(z, p))
    for j in range(len(id_manifold_max)):
        z = (z_manifold_res[id_manifold_max[j] - width],
             z_manifold_res[id_manifold_max[j] + width])
        p = (p_manifold_res[id_manifold_max[j] - width],
             p_manifold_res[id_manifold_max[j] + width])
        max_tangent_coeff.append(linear_coefficients(z, p))
    min_tangent_coeff = np.asarray(min_tangent_coeff)
    max_tangent_coeff = np.asarray(max_tangent_coeff)
    width = 0.02
    # for j in range(len(min_tangent_coeff)):
    #     x_min = z_manifold_res[id_manifold_min[j]]
    #     x = np.array((x_min - width * 0.5, x_min + width * 0.5))
    #     y = polyval(x, min_tangent_coeff[j])
    #     ax1.plot(x, y)
    # for j in range(len(max_tangent_coeff)):
    #     x_max = z_manifold_res[id_manifold_max[j]]
    #     x = np.array((x_max - width * 0.5, x_max + width * 0.5))
    #     y = polyval(x, max_tangent_coeff[j])
    #     ax1.plot(x, y)
    tangent_coeffs = np.asarray((min_tangent_coeff, max_tangent_coeff))
    tangent_coeffs = tangent_coeffs.transpose((2, 0, 1))
    lin_seg_points_min = find_linear_segment_interceptions(tangent_coeffs)
    lin_seg_interceptions.append(lin_seg_points_min)
    tangent_coeffs = \
        np.asarray((min_tangent_coeff[1:], max_tangent_coeff[:-1]))
    # tangent_coeffs = \
    #     np.asarray((min_tangent_coeff[:-1], max_tangent_coeff[1:]))
    tangent_coeffs = tangent_coeffs.transpose((2, 0, 1))
    lin_seg_points_max = \
        find_linear_segment_interceptions(tangent_coeffs)
    lin_seg_points = \
        np.concatenate((lin_seg_points_min, lin_seg_points_max), axis=1)
    id_sort = lin_seg_points[0].argsort()
    lin_seg_points = \
        lin_seg_points[:, id_sort]
    ax1.plot(lin_seg_points[0], lin_seg_points[1])
    plt.show()
    # dp_junction_3.append(lin_seg_points_max[1] - lin_seg_points_min[1])
    # zeta_junction_3.append((2.0 * dp_junction_3[i] / props.density
    #                         - (v2[i] ** 2.0 - v1[i] ** 2.0)
    #                         * geom.manifold_flow_direction[i][-1])
    #                        / (v1[i] ** 2.0))

# zeta_junction_2 = 2.0 * dp_junction_2 / density \
#     / (v1[:-1] ** 2.0)
    velocity_ratio.append(v2[i]/v1[i])

    if geom.manifold_flow_direction[i][-1] == 1:
        zeta_junction_idelchik.append(0.4 * (1.0 - velocity_ratio[i] ** 2.0))
    else:
        # function must be implemented here from idelchik
        zeta_junction_idelchik.append(np.zeros(zeta_junction.shape[-1]))
    
    # free fitting similar to idelchik model
    zeta_junction_fit.append(0.4 * (1.0 - velocity_ratio[i] ** 2.0))

mfd_id = 0
# pressure due to changes in dynamic pressure in manifold 1
dp_dyn = calc_pressure_drop(manifold_velocity[mfd_id], props.density, -0.01,
                            geom.manifold_flow_direction[mfd_id][-1])
print(dp_dyn)
p_dyn = np.zeros(manifold_velocity[mfd_id].shape)
p_dyn[:] = p_manifold[mfd_id][-1]
print(p_manifold_function[mfd_id](z_junction_in[0]))
pressure_direction = -geom.manifold_flow_direction[mfd_id][-1]
add_source(p_dyn, dp_dyn, direction=pressure_direction)
z_dyn = \
    np.append(z_junction_in, z_junction_in[-1] + geom.channel_distance_z)

# p_total = p_manifold[mfd_id] + 0.5 * density * manifold_velocity

dpi = 200
figsize = (6.4 * 2.0, 4.8 * 2.0)
fig = plt.figure(dpi=dpi, figsize=figsize)
plt.plot(z_junction, zeta_junction[mfd_id], 'k.')
plt.plot(z_junction, zeta_junction_2[mfd_id], 'r.')
plt.plot(z_junction, zeta_junction_idelchik[mfd_id], 'b.')
# plt.plot(z_junction, zeta_junction_3[mfd_id], 'g.')
plt.show()
plt.savefig(os.path.join(output_dir_name, 'inlet_x_zeta_junction_manifold.png'))

fig = plt.figure(dpi=dpi, figsize=figsize)
plt.plot(velocity_ratio[mfd_id], zeta_junction_2[mfd_id], 'k.')
# plt.plot(velocity_ratio[mfd_id], zeta_junction_idelchik[mfd_id], 'b.')
# plt.show()
plt.savefig(os.path.join(output_dir_name, 'inlet_zeta_junction_manifold.png'))

fig, ax1 = plt.subplots(dpi=dpi, figsize=figsize)
z_plot = z_manifold[mfd_id]
xticks = np.arange(z_plot[0], z_plot[-1], 0.005)
ax1.plot(z_plot, p_manifold[mfd_id])
ax1.plot(z_plot, p_manifold_function[mfd_id](z_plot))
ax1.plot(z_dyn, p_dyn)
ax2 = ax1.twinx()
# ax2.set_xticks(xticks)
ax1.xaxis.set_major_locator(MultipleLocator(0.01))
#ax1.xaxis.set_minor_locator(MultipleLocator(0.0025))
# ax2 = plt.gca()
ax1.grid(True, which='major')
ax2.plot(z_manifold[mfd_id], junction_square)
# plt.show()

plt.savefig(os.path.join(output_dir_name, 'inlet_manifold_pressure.png'))

channel_data = np.load(channel_file_name)
y_channel = np.asarray([channel_data[3 * i]
                        for i in range(geom.n_channels)])
z_channel = np.asarray([channel_data[3 * i + 1]
                        for i in range(geom.n_channels)])
p_channel = np.asarray([channel_data[3 * i + 2]
                        for i in range(geom.n_channels)])
lin_coeffs = \
    [np.polynomial.polynomial.polyfit(y_channel[i][100:-100],
                                      p_channel[i][100:-100], 1)
     for i in range(geom.n_channels)]

# chl_id = [0, 10, 20, 30, 39]

y_channel_in = geom.manifold_y[0] - geom.manifold_diameter * 0.5
p_channel_linear_in = \
    [polyval(y_channel_in, lin_coeffs[i]) for i in range(geom.n_channels)]
y_channel_out = geom.manifold_y[0] + geom.manifold_diameter * 0.5
p_channel_linear_out = \
    [polyval(y_channel_out, lin_coeffs[i]) for i in range(geom.n_channels)]
dp_junction_channel_in = [p_channel_linear_in[i] - p_junction_in[1][i]
                          for i in range(geom.n_channels)]

# fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
# x = lin_seg_interceptions[mfd_id][0]
# ax.plot(x, lin_seg_interceptions[mfd_id][1], linestyle='-')
# ax.plot(x, p_channel_linear_in, linestyle=':')
# plt.show()

fig = plt.figure(dpi=dpi, figsize=figsize)
# colors = ['k', 'b', 'r', 'g', 'm']
for i in range(geom.n_channels):
    plt.plot(y_channel[i], p_channel[i], linestyle='-')
    plt.plot(y_channel[i], polyval(y_channel[i], lin_coeffs[i]),
             linestyle=':')
plt.show()

fig = plt.figure(dpi=dpi, figsize=figsize)
# colors = ['k', 'b', 'r', 'g', 'm']
y_channel_range = y_channel[:, :100]
p_channel_range = p_channel[:, :100]
for i in range(geom.n_channels):
    plt.plot(y_channel_range[i], p_channel_range[i], linestyle='-')
    plt.plot(y_channel_range[i], polyval(y_channel_range[i], lin_coeffs[i]),
             linestyle=':')
plt.show()

fig = plt.figure(dpi=dpi, figsize=figsize)
colors = ['k', 'b', 'r', 'g', 'm']
y_channel_range = y_channel[:, -100:]
p_channel_range = p_channel[:, -100:]
for i in range(geom.n_channels):
    plt.plot(y_channel_range[i], p_channel_range[i], 
             linestyle='-')
    plt.plot(y_channel_range[i], polyval(y_channel_range[i], lin_coeffs[i]),
             linestyle=':')
plt.show()

print('Last channel x-position:')
print(y_channel[-1, 0])
print('Last channel inlet pressure:')
print(p_channel[-1, 0])
print('Last channel manifold pressure:')
print(p_manifold_function[1](y_channel[-1, 0]))
