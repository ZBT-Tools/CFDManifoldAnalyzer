import numpy as np
import math
import pandas as pd
from scipy.interpolate import griddata, interp1d
from scipy import signal
import os
import re
from ..settings import file_names
from ..settings import geometry as geom
from .output import OutputObject
from . import constants
from abc import ABC, abstractmethod


def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


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

        # assure channel is aligned with axis of coordinate system
        if len(self.direction_vector[self.direction_vector != 0]) != 1:
            raise ValueError('channel direction must be aligned with '
                             'coordinate axis: self.direction_vector must only '
                             'have one non-zero value')

        self.dx = dx
        self.nx = int(np.round(self.length / self.dx)) + 1
        self.data = {key: np.zeros(self.nx) for key in value_names}
        for name in value_names:
            setattr(self, name, np.zeros(self.nx))
        self.x = np.zeros(self.nx)
        self.coords = np.zeros((self.dims, self.nx))
        self.cord_length = np.linspace(0.0, self.length, self.nx)
        self.inlet_mass_flow = 0.0
        self.data_function = {key: None for key in value_names}

    def process(self):
        pass

    def make_interpolation_function(self, data_name='pressure',
                                    method='linear'):
        self.data_function[data_name] = \
            interp1d(self.x, self.data[data_name], kind=method)

    def create_coords(self):
        length_vector = self.length * self.direction_vector
        end_vector = self.start_vector + length_vector
        coords = \
            np.asarray([np.linspace(self.start_vector[i], end_vector[i],
                                    self.nx)
                        for i in range(len(end_vector))])
        self.coords[:] = coords
        self.x[:] = np.dot(self.coords.transpose(), self.direction_vector)
        return coords

    def plot(self, x=None, y=None, xlabel='Channel Coordinate [m]',
             ylabel='Pressure [Pa]', data_name='pressure',
             colormap=None, ax=None, **kwargs):
        if x is None:
            x = self.x
        if y is None:
            y = self.data[data_name]
        return super().plot_lines(x, y, xlabel=xlabel, ylabel=ylabel,
                                  colormap=colormap, ax=ax, **kwargs)


class LinearCFDDataChannel(CFDDataChannel):

    def __init__(self, diameter, length, start_vector, direction_vector, dx,
                 lin_segments=None, value_names=('pressure',), name=None):
        super().__init__(diameter, length, start_vector, direction_vector, dx,
                         value_names=value_names, name=name)
        # if not isinstance(lin_segments, (tuple, list, np.ndarray)):
        #     raise TypeError('argument lin_segments must be iterable with each'
        #                     ' entry containing a coordinate pair (start, end) '
        #                     'of a channel segment to be linearized')
        self.lin_segments = lin_segments
        self.lin_coeffs = None

    @staticmethod
    def linear_coefficients(x, y, method='2-points'):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError('x and y must have equal shapes')
        if np.ndim(x) != 1 or x.shape[-1] < 2:
            raise ValueError('x and y must be one-dimensional array with at '
                             'least two entries')
        if method == '2-points' or x.shape[-1] == 2:
            m = (y[-1] - y[0]) / (x[-1] - x[0])
            b = y[0] - m * x[0]
            return np.asarray((b, m))
        elif method == 'polyfit':
            return np.polynomial.polynomial.polyfit(x, y, 1)
        else:
            raise NotImplementedError

    def calc_linear_coefficients(self, lin_segments=None, data_name='pressure'):
        lin_seg_describer = 'lin_segments must be iterable with coordinate ' + \
                            'pairs (start, end) describing each ' + \
                            'linear segment of interest for the channel'
        if lin_segments is None:
            if self.lin_segments is None:
                raise ValueError('either argument lin_segments or object '
                                 'attribute self.lin_segments must not be None')
            else:
                lin_segments = self.lin_segments
        lin_segments = np.asarray(lin_segments)

        if lin_segments.shape[-1] != 2:
            raise ValueError(lin_seg_describer)
        if lin_segments.shape == (1, 1):
            n_segs = 1
            lin_segments = np.asarray([lin_segments])
        elif np.dim(lin_segments) == 2:
            n_segs = len(lin_segments)
        else:
            raise ValueError(lin_seg_describer)
        lin_coeffs = []
        for seg in lin_segments:
            idx0 = find_nearest_idx(self.x, seg[0])
            idx1 = find_nearest_idx(self.x, seg[1])

            lin_coeffs.append(
                self.linear_coefficients(self.x[idx0:idx1],
                                         self.data[data_name][idx0:idx1]))
        lin_coeffs = np.asarray(lin_coeffs)
        return lin_coeffs

    def set_linear_coefficients(self, lin_segments=None, data_name='pressure'):
        self.lin_coeffs = self.calc_linear_coefficients(lin_segments, data_name)

    def get_linear_coefficients(self, lin_segments=None, data_name='pressure'):
        if self.lin_coeffs is None:
            self.set_linear_coefficients(lin_segments, data_name)
        return self.lin_coeffs

    def linear_values(self, x, lin_segment=None, data_name='pressure'):
        x = np.asarray(x)
        if lin_segment is None:
            if self.lin_segments is not None:
                lin_segment = self.lin_segments
            else:
                raise ValueError('either argument lin_segment or object '
                                 'attribute self.lin_segments must not be None')

        lin_segment = np.asarray(lin_segment)
        if x.ndim != lin_segment.ndim:
            raise ValueError('arguments x and lin_segment '
                             'must have equal dimensions')
        if x.ndim == 1:
            lin_coeffs = self.calc_linear_coefficients(lin_segment, data_name)
            return np.polynomial.polynomial.polyval(x, lin_coeffs)
        elif x.ndim == 2:
            lin_coeffs = self.get_linear_coefficients(lin_segment, data_name)
            return np.asarray(
                [np.polynomial.polynomial.polyval(x[i], lin_coeffs[i])
                 for i in range(len(x))])


class ManifoldCFDDataChannel(LinearCFDDataChannel):

    def __init__(self, diameter, length, start_vector, direction_vector, dx,
                 x_range, value_names=('pressure',), name=None):
        super().__init__(diameter, length, start_vector, direction_vector, dx,
                         value_names=value_names, name=name)
        if len(x_range) != 2:
            raise TypeError('argument x_range must be iterable with two '
                            'entries (start, end) of the coordinate range to '
                            'be analyzed')
        # best guess parameters for filtering operations
        self.resolution = 1000
        self.poly_order = 5
        self.window_length = np.int(np.round(self.resolution/10) // 2 * 2 + 1)
        self.x_range = np.linspace(x_range[0], x_range[1], self.resolution)
        self.data_range = {key: np.zeros(self.x_range.shape)
                           for key in self.data.keys()}

    def calculate_data_range(self, data_name='pressure'):
        if self.data_function[data_name] is None:
            self.make_interpolation_function(data_name)
        self.data_range[data_name] = self.data_function[data_name](self.x_range)

    def calculate_gradient(self, data, order=1, filter_data=True):
        """
        :param data: 1D numpy array to manipulate
        :param order: order of gradient/derivative
        :param filter_data: filter/smooth (with savgol filter) data between each
        manipulation step
        :param data_name: channel data to be manipulated
        :return: 1D-array with local minima of higher order derivatives
        """
        if filter_data is True:
            data = signal.savgol_filter(data, window_length=self.window_length,
                                        polyorder=self.poly_order)
        grad_data = data
        for i in range(order):
            grad_data = np.gradient(grad_data)
            if filter_data is True:
                grad_data = \
                    signal.savgol_filter(grad_data,
                                         window_length=self.window_length,
                                         polyorder=self.poly_order)
        return grad_data

    def get_higher_order_min_id(self, order=1, filter_data=True,
                                data_name='pressure', init_data=True):
        if init_data:
            self.calculate_data_range(data_name)
        grad_data, data = self.calculate_gradient(self.data_range[data_name],
                                                  order=order,
                                                  filter_data=filter_data)
        id_min = signal.argrelmin(grad_data, order=5)[0]
        return np.array((self.x_range[id_min], data[id_min]))

    def get_higher_order_max_id(self, order=1, filter_data=True,
                                data_name='pressure', init_data=True):
        if init_data:
            self.calculate_data_range(data_name)
        grad_data, data = self.calculate_gradient(self.data_range, order=order,
                                                  filter_data=filter_data)
        return signal.argrelmax(grad_data, order=5)[0]

    def get_higher_order_min(self, order=1, filter_data=True,
                             data_name='pressure', init_data=True):
        id_min = self.get_higher_order_min_id(order=order,
                                              filter_data=filter_data,
                                              data_name=data_name,
                                              init_data=init_data)
        return np.array((self.x_range[id_min],
                         self.data_range[data_name][id_min]))

    def get_higher_order_max(self, order=1, filter_data=True,
                             data_name='pressure', init_data=True):
        id_max = self.get_higher_order_max_id(order=order,
                                              filter_data=filter_data,
                                              data_name=data_name,
                                              init_data=init_data)
        return np.array((self.x_range[id_max],
                         self.data_range[data_name][id_max]))

    @staticmethod
    def calc_linear_interceptions(coeffs):
        coeffs = np.asarray(coeffs)
        if not coeffs.shape[0] == 2 and not coeffs.shape[0] == 2:
            raise ValueError(
                'parameter coeffs must be numpy array with the first '
                'two dimensions of shape (2,2)')
        m = coeffs[1]
        b = coeffs[0]
        x = (b[1] - b[0]) / (m[0] - m[1])
        y = m[0] * x + b[0]
        return np.array((x, y))

    def calc_linear_coeffs_from_id(self, arg_id, id_width=1,
                                   data_name='pressure'):
        linear_coeffs = []
        for i in range(len(arg_id)):
            x = np.asarray(self.x_range[arg_id[i] - id_width],
                           self.x_range[arg_id[i] + id_width])
            y = np.asarray(self.data_range[data_name][arg_id[i] - id_width],
                           self.data_range[data_name][arg_id[i] + id_width])
            linear_coeffs.append(self.linear_coefficients(x, y))
        return np.asarray(linear_coeffs)

    def calc_linear_segment_interceptions(self, id_width=1, order='minmax',
                                          data_name='pressure'):
        grad_order = 1
        id_linear_min = self.get_higher_order_min(order=grad_order)
        id_linear_max = self.get_higher_order_max(order=grad_order)
        linear_min_coeffs = \
            self.calc_linear_coeffs_from_id(id_linear_min, id_width=id_width,
                                            data_name=data_name)
        linear_max_coeffs = \
            self.calc_linear_coeffs_from_id(id_linear_max, id_width=id_width,
                                            data_name=data_name)
        if order == 'minmax':
            linear_coeffs = np.asarray((linear_min_coeffs, linear_max_coeffs))
        elif order == 'maxmin':
            linear_coeffs = np.asarray((linear_max_coeffs, linear_min_coeffs))
        else:
            raise ValueError('parameter order must be "minmax" or "maxmin"')
        linear_coeffs = linear_coeffs.transpose((2, 0, 1))
        return self.calc_linear_interceptions(linear_coeffs)


class CFDMassFlowProcessor(OutputObject, ABC):

    def __new__(cls, data, output_dir, name=None, **kwargs):

        if isinstance(data, (list, tuple, np.ndarray)):
            return super(CFDMassFlowProcessor,
                         cls).__new__(CFDMassFlowArrayProcessor)
        elif os.path.isfile(data):
            return super(CFDMassFlowProcessor,
                         cls).__new__(CFDMassFlowFileProcessor)

        else:
            raise NotImplementedError

    def __init__(self, data, output_dir, name=None, **kwargs):
        super().__init__(name)
        self.output_dir = output_dir
        self.n_channels = None
        self.mass_flows = None
        self.total_mass_flow = 0.0

    @abstractmethod
    def process(self, total_mass_flow=None):
        pass

    def save(self, name='mass_flow_data', as_ascii=True):
        if as_ascii:
            np.savetxt(os.path.join(self.output_dir, name), self.mass_flows)
        else:
            np.save(os.path.join(self.output_dir, name), self.mass_flows)

    def plot(self, xlabel='Channels [-]', ylabel='Normalized Mass Flow [-]',
             colormap=None, ax=None, **kwargs):
        x = np.array([i for i in range(self.n_channels)])
        y = self.mass_flows / np.mean(self.mass_flows)
        return super().plot_lines(x, y, xlabel=xlabel, ylabel=ylabel,
                                  colormap=colormap, ax=ax, **kwargs)


class CFDMassFlowFileProcessor(CFDMassFlowProcessor):
    def __init__(self, data, output_dir, name=None, **kwargs):
        super().__init__(output_dir, name)
        self.file_path = data
        self.mass_flow_name = \
            kwargs.pop('flow_key', file_names.mass_flow_name)
        self.total_mass_flow_name = \
            kwargs.pop('total_flow_key', file_names.total_mass_flow_name)

    def load_2d_data(self):
        return pd.read_csv(self.file_path, sep='\t', header=[0, 1])

    def process(self, total_mass_flow=None):
        cfd_data_2d = self.load_2d_data()
        self.mass_flows = \
            cfd_data_2d[self.mass_flow_name].iloc[-2].to_numpy()
        self.n_channels = len(self.mass_flows)
        if self.total_mass_flow is None:
            self.total_mass_flow = \
                cfd_data_2d[self.total_mass_flow_name].iloc[-2][0]
        else:
            self.total_mass_flow = total_mass_flow
        return self.mass_flows


class CFDMassFlowArrayProcessor(CFDMassFlowProcessor):
    def __init__(self, data, output_dir, name=None, **kwargs):
        super().__init__(output_dir, name)
        self.mass_flows = np.asarray(data)

    def process(self, total_mass_flow=None):
        self.n_channels = len(self.mass_flows)
        if self.total_mass_flow is None:
            self.total_mass_flow = np.sum(self.mass_flows)
        else:
            self.total_mass_flow = total_mass_flow
        return self.mass_flows


class CFDManifoldProcessor(OutputObject):
    def __init__(self, pressure_file_path, mass_flow_data,
                 output_dir, name=None):
        super().__init__(name)
        self.pressure_file_path = pressure_file_path
        self.output_dir = output_dir
        # create output folder
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.collections = ('channel', 'manifold')

        self.n_channels = geom.n_channels
        self.n_manifolds = geom.n_manifolds

        # create channel data objects
        # assure correct dimensions of flow direction array
        channel_flow_direction = np.asarray(geom.channel_flow_direction)
        channel_start_vector = np.asarray(geom.channel_start_vector)
        if channel_flow_direction.shape != (self.n_channels, 3):
            raise ValueError('shape of channel_flow_direction must be '
                             'two-dimensional array with each sub-array '
                             'having three values')
        self.channels = []
        for i in range(geom.n_channels):
            start_vector = channel_start_vector[i]
            direction_vector = channel_flow_direction[i]
            self.channels.append(
                LinearCFDDataChannel(geom.channel_diameter, geom.channel_length,
                                     start_vector, direction_vector,
                                     geom.channel_dy, geom.lin_segments))

        # create manifold data objects
        manifold_flow_direction = np.asarray(geom.manifold_flow_direction)
        manifold_start_vector = np.asarray(geom.manifold_start_vector)
        if manifold_flow_direction.shape != (self.n_manifolds, 3):
            raise ValueError('shape of manifold_flow_direction must be '
                             'two-dimensional array with each sub-array '
                             'having three values')
        self.manifolds = []
        for i in range(geom.n_manifolds):
            start_vector = manifold_start_vector[i]
            direction_vector = manifold_flow_direction[i]
            self.manifolds.append(ManifoldCFDDataChannel(
                geom.manifold_diameter, geom.manifold_length, start_vector,
                direction_vector, geom.manifold_dz,
                geom.manifold_range))
        # initialize mass flow data
        self.mass_flow_data = \
            CFDMassFlowProcessor(mass_flow_data, self.output_dir)

        # status flag if data has been processed
        self.is_processed = False

    def process(self, **kwargs):
        self.interpolate_data()
        self.make_interpolation_functions()
        total_mass_flow = kwargs.get('total_mass_flow', None)
        self.mass_flow_data.process(total_mass_flow=total_mass_flow)
        self.is_processed = True

    def load_3d_data(self):
        file_ext = self.pressure_file_path.split('.')[-1]
        if file_ext == 'npy':
            return np.load(self.pressure_file_path).transpose()
        elif file_ext == 'dat':
            return np.loadtxt(self.pressure_file_path).transpose()
        else:
            raise IOError('file must be of ascii (.dat) or binary (.npy) '
                          'format')

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
        pressure_sum = np.sum(pressure_combined)
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
            chl.make_interpolation_function()
        for mfd in self.manifolds:
            mfd.make_interpolation_function()

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

    def plot_collection(self, collection_name, data_name='pressure',
                        xlabel='Coordinate [m]', ylabel='Pressure [Pa]',
                        **kwargs):
        if collection_name == 'channel':
            collection = self.channels
        elif collection_name == 'manifold':
            collection = self.manifolds
        else:
            raise ValueError('collection with name {}'
                             ' not available'.format(collection_name))
        # plot channel pressures
        x = [channel.x for channel in collection]
        y = [channel.data[data_name] for channel in collection]
        file_path = os.path.join(self.output_dir,
                                 collection_name + '_' + data_name)
        if 'name_extension' in kwargs:
            file_path += kwargs['name_extension']
        file_path += '.png'
        self.create_figure(file_path, x, y, xlabels=xlabel, ylabels=ylabel,
                           marker=None, **kwargs)

    def plot(self, data_name='pressure', ylabel='Pressure [Pa]', **kwargs):
        # plot channel pressures
        self.plot_collection('channel', data_name=data_name, ylabel=ylabel,
                             **kwargs)
        # plot manifold pressures
        self.plot_collection('manifold', data_name=data_name, ylabel=ylabel,
                             **kwargs)

        # plot mass flow distribution
        x = np.array([i for i in range(self.n_channels)])
        mass_flows = self.mass_flow_data.mass_flows
        y = mass_flows / np.mean(mass_flows)
        file_path = os.path.join(self.output_dir, 'mass_flow_distribution.png')
        self.create_figure(file_path, x, y, xlabels='Channels [-]',
                           ylabels='Normalized Mass Flow [-]', **kwargs)


class CFDTJunctionProcessor(CFDManifoldProcessor):
    def __init__(self, pressure_file_path, mass_flow_data,
                 output_dir, name=None):
        super().__init__(pressure_file_path, mass_flow_data,
                         output_dir, name)
        self.coordinate_name = "-coordinate"
        self.velocity_name = "-velocity"
        self.coordinates = ('x', 'y', 'z')

    def load_3d_data(self, processor_split=True, pattern='proc_'):
        path = os.path.abspath(self.pressure_file_path)
        dir_name = os.path.dirname(path)
        if processor_split:
            if pattern in self.pressure_file_path:
                # find all processor-divided pressure files
                dir_entries = os.listdir(os.path.dirname(path))
                pressure_files = \
                    [os.path.join(dir_name, entry) for entry in dir_entries
                     if pattern in entry]
                # get column names from first file
                column_names = pd.read_csv(pressure_files[0], sep=', |,',
                                           header=[1], nrows=0).columns.tolist()
                # replace column names for coordinates and velocity
                coordinate_ids = (1, 2, 3)
                velocity_ids = (7, 8, 9)
                for i in range(len(self.coordinates)):
                    column_names[coordinate_ids[i]] = \
                        self.coordinates[i] + self.coordinate_name
                    column_names[velocity_ids[i]] = \
                        self.coordinates[i] + self.velocity_name
                # load all files as dataframes
                df_list = [pd.read_csv(file, header=None, sep='\s+',
                                       skiprows=[0, 1], names=column_names)
                           for file in pressure_files]
                # concatenate all dfs
                return pd.concat(df_list, ignore_index=True)
            else:
                raise ValueError('pattern "{}" not found in file '
                                 '"{}"'.format(pattern,
                                               self.pressure_file_path))
        else:
            raise NotImplementedError

    def data_to_array(self):
        df = self.load_3d_data()
        # create coordinate and pressure arrays corresponding to coordinate
        # system configuration in AVL FIRE case setup
        coordinates = []
        for i in range(len(self.coordinates)):
            coordinate = \
                df[self.coordinates[i] + self.coordinate_name].to_numpy()
            coordinates.append(coordinate)
        coord_array = np.asarray(coordinates)
        value_array = df['pressure'].to_numpy()
        combined_array = np.asarray(np.vstack((coord_array, value_array)))
        return combined_array, coord_array.transpose(), value_array

    def plot(self, data_name='pressure', ylabel='Pressure [Pa]', **kwargs):
        # plot channel pressures
        self.plot_collection('channel', data_name=data_name, ylabel=ylabel,
                             **kwargs)
        # plot manifold pressures
        self.plot_collection('manifold', data_name=data_name, ylabel=ylabel,
                             **kwargs)
