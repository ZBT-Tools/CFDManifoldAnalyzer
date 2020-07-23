import numpy as np
import math
import pandas as pd
from scipy.interpolate import griddata, interp1d
from scipy import signal
import os
import timeit
from ..settings import file_names
from ..settings import geometry as geom
from .output import OutputObject
from . import constants


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
        self.x[:] = np.dot(self.coords, self.direction_vector)
        return coords

    def plot(self, x=None, y=None, xlabel='Channel Coordinate [m]',
             ylabel='Pressure [Pa]', data_name='pressure',
             colormap=None, ax=None, **kwargs):
        if x is None:
            x = self.cord_length
        if y is None:
            y = self.data[data_name]
        return super().plot_lines(x, y, xlabel=xlabel, ylabel=ylabel,
                                  colormap=colormap, ax=ax, **kwargs)


class LinearCFDDataChannel(CFDDataChannel):

    def __init__(self, diameter, length, start_vector, direction_vector, dx,
                 lin_segments=None, value_names=('pressure',), name=None):
        super().__init__(diameter, length, start_vector, direction_vector, dx,
                         value_names=value_names, name=name)
        if not isinstance(lin_segments, (tuple, list, np.ndarray)):
            raise TypeError('argument lin_segments must be iterable with each'
                            ' entry containing a coordinate pair (start, end) '
                            'of a channel segment to be linearized')
        self.lin_segments = lin_segments
        self.lin_coeffs = None

    @staticmethod
    def _linear_coefficients(x, y, method='2-points'):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError('x and y must have equal shapes')
        if np.dim(x) != 1 or x.shape[-1] < 2:
            raise ValueError('x and y must be one-dimensional array with at '
                             'least two entries')
        if method == '2-points' or x.shape[-1] == 2:
            m = (y[2] - y[1]) / (x[2] - x[1])
            b = y[1] - m * x[1]
            return np.asarray((b, m))
        elif method == 'polyfit':
            np.polynomial.polynomial.polyfit(x, y, 1)
        else:
            raise NotImplementedError

    def calc_linear_coefficients(self, lin_segments=None, data_name='pressure'):
        lin_seg_describer = 'lin_segments must be iterable with coordinate ' + \
                            'pairs (start, end) describing each ' + \
                            'linear segment of interest for the channel'
        if lin_segments is None:
            if self.lin_segments is None:
                raise ValueError('either argument lin_segments nor object '
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
                self._linear_coefficients(self.x[idx0:idx1],
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


class ManifoldCFDDataChannel(CFDDataChannel):

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

    def calculate_gradient(self, order=2, filter_data=True,
                           data_name='pressure'):
        """
        :param order: order of gradient/derivative
        :param filter_data: filter/smooth (with savgol filter) data between each
        manipulation step
        :param data_name: channel data to be manipulated
        :return: 1D-array with local minima of higher order derivatives
        """
        if self.data_function[data_name] is None:
            self.make_interpolation_function(data_name)

        data = self.data_function[data_name](self.x_range)
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
        return grad_data, data

    def get_higher_order_min(self, order=2, filter_data=True,
                             data_name='pressure'):
        grad_data, data = self.calculate_gradient(order=order,
                                                  filter_data=filter_data,
                                                  data_name=data_name)
        return data[signal.argrelmin(grad_data, order=5)[0]]

    def get_higher_order_max(self, order=2, filter_data=True,
                             data_name='pressure'):
        grad_data, data = self.calculate_gradient(order=order,
                                                  filter_data=filter_data,
                                                  data_name=data_name)
        return data[signal.argrelmax(grad_data, order=5)[0]]

    def get_higher_order_minmax(self, order=2, filter_data=True,
                             data_name='pressure'):
        grad_data, data = self.calculate_gradient(order=order,
                                                  filter_data=filter_data,
                                                  data_name=data_name)
        data_min = data[signal.argrelmin(grad_data, order=5)[0]]
        data_max = data[signal.argrelmax(grad_data, order=5)[0]]
        return np.asarray((data_min, data_max))


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

        self.n_channels = geom.n_channels
        self.n_manifolds = geom.n_manifolds

        # create channel data objects
        # assure correct dimensions of flow direction array
        channel_flow_direction = np.asarray(geom.channel_flow_direction)
        if channel_flow_direction.shape != (3,):
            raise ValueError('shape of channel_flow_direction must be '
                             'one-dimensional array with three values')
        self.channels = []
        for i in range(geom.n_channels):
            start_vector = \
                np.asarray((0.0,
                            geom.manifold_y[0] - 0.5 * geom.manifold_diameter,
                            geom.channel_0_z + i * geom.channel_distance_z))
            direction_vector = channel_flow_direction
            self.channels.append(
                LinearCFDDataChannel(geom.channel_diameter, geom.channel_length,
                                     start_vector, direction_vector,
                                     geom.channel_dy, geom.lin_segments))

        # create manifold data objects
        manifold_flow_direction = np.asarray(geom.manifold_flow_direction)
        if manifold_flow_direction.shape != (self.n_manifolds, 3):
            raise ValueError('shape of channel_flow_direction must be '
                             'two-dimensional array with three values for '
                             'each manifold')
        self.manifolds = []
        for i in range(geom.n_manifolds):
            mfd_flow_dir = manifold_flow_direction[i]
            if mfd_flow_dir[mfd_flow_dir != 0] > 0:
                idz = 0
            elif mfd_flow_dir[mfd_flow_dir != 0] < 0:
                idz = -1
            else:
                raise ValueError('each manifold flow direction vector must '
                                 'contain a single non-zero positive or '
                                 'negative value aligned with a coordinate '
                                 'axis')
            start_vector = np.asarray((0.0, geom.manifold_y[i],
                                       geom.bounding_box[-1, idz]))
            direction_vector = np.asarray(mfd_flow_dir)
            self.manifolds.append(ManifoldCFDDataChannel(
                geom.manifold_diameter, geom.manifold_length, start_vector,
                direction_vector, geom.manifold_dz,
                geom.manifold_range))
        # initialize mass flow data
        self.mass_flow_data = CFDMassFlowProcessor(self.mass_flow_file_path,
                                                   self.output_dir)

        # status flag if data has been processed
        self.is_processed = False

    def process(self):
        self.interpolate_data()
        self.make_interpolation_functions()
        self.mass_flow_data.process()
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
            chl.data_function[data_name] = interp1d(chl.x, chl.data[data_name])
        for mfd in self.manifolds:
            mfd.data_function[data_name] = interp1d(mfd.x, mfd.data[data_name])

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



