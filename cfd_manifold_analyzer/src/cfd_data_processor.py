import numpy as np
import pandas as pd
from scipy.interpolate import griddata, interp1d
import os
import timeit
from ..settings import file_names
from ..settings import geometry as geom
from .output import OutputObject
from . import constants


class CFDDataChannel(OutputObject):
    def __init__(self, diameter, length, start_vector, direction_vector, dx,
                 value_names=('pressure',), name=None):
        super().__init__(name)
        self.dims = 3
        self.diameter = diameter
        self.length = length
        start_vector = np.asarray(start_vector)
        direction_vector = np.asarray(direction_vector)
        if np.asarray(start_vector).shape[0] != self.dims:
            raise ValueError(
                'start_vector must be of size {}'.format(self.dims))
        self.start_vector = start_vector
        if np.asarray(direction_vector).shape[0] != self.dims:
            raise ValueError(
                'direction_vector must be of size {}'.format(self.dims))
        # normalize direction vector
        try:
            self.direction_vector = \
                direction_vector / np.linalg.norm(direction_vector)
        except FloatingPointError:
            raise FloatingPointError('direction vector must not be zero')
        self.dx = dx
        self.nx = int(np.round(self.length / self.dx)) + 1
        self.data = {key: np.zeros(self.nx) for key in value_names}
        for name in value_names:
            setattr(self, name, np.zeros(self.nx))
        self.x = np.zeros(self.nx)
        self.coords = np.zeros((self.dims, self.nx))
        self.cord_length = np.linspace(0.0, self.length, self.nx)
        self.inlet_mass_flow = 0.0
        self.data_function = {key: [] for key in value_names}

    def create_coords(self):
        length_vector = self.length * self.direction_vector
        end_vector = self.start_vector + length_vector
        coords = \
            np.asarray([np.linspace(self.start_vector[i], end_vector[i],
                                    self.nx)
                        for i in range(len(end_vector))])
        self.coords[:] = coords
        return coords

    def plot(self, x=None, y=None, xlabel='Cord Length [-]',
             ylabel='CFD Data [-]', colormap=None, ax=None, **kwargs):
        if x is None:
            x = self.cord_length
        if y is None:
            y = self.data[0]
        return super().plot_lines(x, y, xlabel=xlabel, ylabel=ylabel,
                                  colormap=colormap, ax=ax, **kwargs)


class CFDMassFlowProcessor(OutputObject):
    def __init__(self, file_path, output_dir, name=None):
        super().__init__(name)
        self.file_path = file_path
        self.output_dir = output_dir
        self.mass_flow_name = file_names.mass_flow_name
        self.total_mass_flow_name = file_names.total_mass_flow_name
        self.n_channels = None
        self.mass_flows = None
        self.total_mass_flow = 0.0

    def load_2d_data(self):
        return pd.read_csv(self.file_path, sep='\t', header=[0, 1])

    def process(self):
        cfd_data_2d = self.load_2d_data()
        self.mass_flows = \
            cfd_data_2d[self.mass_flow_name].iloc[-2].to_numpy()
        self.n_channels = range(len(self.mass_flows))
        self.total_mass_flow = \
            cfd_data_2d[self.total_mass_flow_name].iloc[-2][0]
        return self.mass_flows

    def save(self, name='mass_flow_data', ascii=False):
        if ascii:
            np.savetxt(os.path.join(self.output_dir, name), self.mass_flows)
        else:
            np.save(os.path.join(self.output_dir, name), self.mass_flows)

    def plot(self, xlabel='Channels [-]', ylabel='Normalized Mass Flow [-]',
             colormap=None, ax=None, **kwargs):
        x = np.array([i for i in range(self.n_channels)])
        y = self.mass_flows / np.mean(self.mass_flows)
        return super().plot_lines(x, y, xlabel=xlabel, ylabel=ylabel,
                                  colormap=colormap, ax=ax, **kwargs)


class CFDManifoldProcessor(OutputObject):
    def __init__(self, pressure_file_path, mass_flow_file_path,
                 output_dir, name=None):
        super().__init__(name)
        self.pressure_file_path = pressure_file_path
        self.mass_flow_file_path = mass_flow_file_path
        self.output_dir = output_dir
        # create output folder
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.collections = ('channel', 'manifold')

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
        # initialize mass flow data
        self.mass_flow_data = CFDMassFlowProcessor(self.mass_flow_file_path,
                                                   self.output_dir)
        self.n_channels = len(self.channels)
        self.n_manifolds = len(self.manifolds)

    def process(self):
        self.interpolate_data()
        self.mass_flow_data.process()

    def load_3d_data(self):
        file_ext = self.pressure_file_path.split('.')[-1]
        if file_ext == 'npy':
            return np.load(self.pressure_file_path).transpose()
        elif file_ext == 'dat':
            return np.loadtxt(self.pressure_file_path).transpose()
        else:
            raise IOError('file must be ascii (.dat) or binary (.npy)')

    def data_to_array(self):
        raw_data = self.load_3d_data()
        # create coordinate and pressure arrays corresponding to coordinate
        # system configuration in AVL FIRE case setup
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

    def interpolate_data(self, data_name='pressure'):
        channel_coords = self.create_channel_coords()
        # combine channel coordinates into one array for efficient interpolation
        channel_coords_combined = np.asarray([item for item in channel_coords])
        channel_coords_combined = \
            channel_coords_combined.reshape(-1,
                                            channel_coords_combined.shape[-1])
        manifold_coords = self.create_manifold_coords()
        # combine manifold coordinates into one array
        # for efficient interpolation
        manifold_coords_combined = \
            np.asarray([item for item in manifold_coords])
        manifold_coords_combined = \
            manifold_coords_combined.reshape(-1,
                                             manifold_coords_combined.shape[-1])

        # combine all coordinates into one array for efficient interpolation
        coords_combined = np.concatenate((channel_coords_combined,
                                          manifold_coords_combined), axis=0)
        # get data coordinates and values from raw data file
        combined_array, data_coords, data_values = self.data_to_array()
        # strip zero dimension
        data_coords_transposed = data_coords.transpose()
        non_zero_axis = []
        for i in range(len(data_coords_transposed)):
            if np.var(data_coords_transposed[i]) > constants.SMALL:
                non_zero_axis.append(i)
        stripped_data_coords = np.take(data_coords, non_zero_axis, axis=-1)
        stripped_coords_combined = \
            np.take(coords_combined, non_zero_axis, axis=-1)
        # interpolate centerline (manifold and channels) pressure values
        # channel_pressure = np.zeros((n_channels, ny_channel))
        # for i in range(n_channels):
        pressure_combined = griddata(stripped_data_coords, data_values,
                                     stripped_coords_combined)
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
            channel.data[data_name] = channel_pressure[i]
        for i, manifold in enumerate(self.manifolds):
            manifold.data[data_name] = manifold_pressure[i]

    def make_interpolation_functions(self, data_name='pressure'):
        for chl in self.channels:
            for coord in chl.coords:
                chl.data_function[data_name].append(
                    interp1d(coord, chl.data[data_name]))
        for mfd in self.manifolds:
            for coord in mfd.coords:
                mfd.data_function[data_name].append(
                    interp1d(coord, mfd.data[data_name]))




    def save_collection(self, collection_name):
        if collection_name == 'channel':
            collection = self.channels
        elif collection_name == 'manifold':
            collection = self.manifolds
        else:
            raise ValueError('collection with name {}'
                             ' not available'.format(collection_name))
        collection_data = []
        for i, item in enumerate(collection):
            for j in range(item.dims):
                collection_data.append(item.coords[j])
            for j in item.data:
                collection_data.append(j)
        collection_data = np.asarray(collection_data)
        data_name = collection_name + '_data'
        np.save(os.path.join(self.output_dir, data_name), collection_data)

    def save(self):
        for collection in self.collections:
            self.save_collection(collection)

    def plot(self, **kwargs):
        # plot channel pressures
        x = self.channels[0].cord_length
        y = [channel.data['pressure'] for channel in self.channels]
        file_path = os.path.join(self.output_dir, 'channel_pressure.png')
        self.create_figure(file_path, x, y, xlabels='Length [m]',
                           ylabels='Pressure [Pa]', marker=None, **kwargs)
        # plot manifold pressures
        x = self.manifolds[0].cord_length
        y = [manifold.data['pressure'] for manifold in self.manifolds]
        file_path = os.path.join(self.output_dir, 'manifold_pressure.png')
        self.create_figure(file_path, x, y, xlabels='Length [m]',
                           ylabels='Pressure [Pa]', marker='', **kwargs)

        # plot mass flow distribution
        x = np.array([i for i in range(self.n_channels)])
        mass_flows = self.mass_flow_data.mass_flows
        y = mass_flows / np.mean(mass_flows)
        file_path = os.path.join(self.output_dir, 'mass_flow_distribution.png')
        self.create_figure(file_path, x, y, xlabels='Channels [-]',
                           ylabels='Normalized Mass Flow [-]', **kwargs)



