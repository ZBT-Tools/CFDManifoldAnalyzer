# general imports
import numpy as np
import os
import shutil
import string
import weakref
from copy import deepcopy
from itertools import cycle, islice
import matplotlib
# configure backend here
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# local module imports
from . import constants
from ..settings import output
from ..settings import file_names

# plotting
FONT_SIZE = 14
NUMBER_SIZE = 14
MARKER_SIZE = 5.0
LINE_WIDTH = 1.0
FIG_DPI = 150
FIG_SIZE = (6.4, 4.8)


class OutputObject:

    # PRINT_HIERARCHY = 3
    # CLUSTER_NAMES = [['Cell', 'Flow Circuit']]
    _instances = set()

    def __init__(self, name):
        # assert isinstance(name, str)
        self._name = name
        self.active = True

        self.print_data_1d = {}
        self.print_data_2d = {}
        self.print_data = [self.print_data_1d, self.print_data_2d]
        self._instances.add(weakref.ref(self))

        self.save_csv = output.save_csv
        # switch to save the csv data
        self.save_plot = output.save_plot
        # switch to save the plot data

        self.delimiter = ','
        self.csv_format = '%.9e'
        # object of the class Stack
        self.output_dir = file_names.output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            # shutil.rmtree(self.output_dir, ignore_errors=True)

    def _get_name(self):
        return self._name

    def _set_name(self, name):
        self._name = name

    @property
    def name(self):
        return self._get_name()

    @name.setter
    def name(self, name):
        self._set_name(name)

    def extend_data_names(self, name, prepend=True):
        for i, print_data in enumerate(self.print_data):
            print_data_new_keys = {}
            for key, value in print_data.items():
                if prepend:
                    new_key = name + ' ' + key
                else:
                    new_key = key + ' ' + name
                print_data_new_keys[new_key] = value
                # print_data.pop(key)
            self.print_data[i] = print_data_new_keys
        # print_data_new_keys = {}
        # for key, value in self.print_data_1d.items():
        #     if prepend:
        #         new_key = name + ' ' + key
        #     else:
        #         new_key = key + ' ' + name
        #     print_data_new_keys[new_key] = value
        # self.print_data_1d = print_data_new_keys
        # print_data_new_keys = {}
        # for key, value in self.print_data_2d.items():
        #     if prepend:
        #         new_key = name + ' ' + key
        #     else:
        #         new_key = key + ' ' + name
        #     print_data_new_keys[new_key] = value
        # self.print_data_2d = print_data_new_keys
        # self.print_data = [self.print_data_1d, self.print_data_2d]

    @classmethod
    def getinstances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    def copy(self):
        copy = deepcopy(self)
        self._instances.add(weakref.ref(copy))
        return copy

    def add_print_data(self, data_array, name, units='-', sub_names=None):
        if data_array.ndim == 2:
            if sub_names is None:
                sub_names = [str(i+1) for i in range(len(data_array))]
            self.print_data_2d[name] = \
                {sub_names[i]:
                 {'value': data_array[i], 'units': str(units), 'save': True}
                 for i in range(len(sub_names))}
        elif data_array.ndim == 1:
            self.print_data_1d[name] = \
                {'value': data_array, 'units': str(units), 'save': True}
        else:
            raise ValueError('argument data_array must be 1- or 2-dimensional')

    def add_print_variables(self, print_variables):
        for i, name in enumerate(print_variables['names']):
            attr = eval('self.' + name)
            description = string.capwords(name.replace('_', ' '))
            units = print_variables['units'][i]
            sub_names = print_variables.get('sub_names', None)
            if sub_names is not None:
                sub_names = eval(sub_names[i])
            self.add_print_data(attr, description, units=units,
                                sub_names=sub_names)

    @staticmethod
    def combine_print_variables(dict_a, dict_b):
        if dict_b is not None:
            for key in dict_a.keys():
                dict_a[key] += dict_b[key]
        return dict_a

    @classmethod
    def make_name_list(cls):
        name_list = []
        for obj in cls.getinstances():
            obj.name_list = obj.name.split(': ')
            name_list.append(obj.name_list)
        return name_list

    @staticmethod
    def clean_directory(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
            except Exception as e:
                print(e)

    @staticmethod
    def set_ax_properties(ax, **kwargs):
        fontsize = kwargs.get('fontsize', FONT_SIZE)
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'], fontsize=fontsize)
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'], fontsize=fontsize)
        if 'margins' in kwargs:
            ax.margins(x=kwargs['margins'][0], y=kwargs['margins'][1])
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'])
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'])
        if 'xticks' in kwargs:
            ax.set_xticks(kwargs['xticks'])
        if 'yticks' in kwargs:
            ax.set_yticks(kwargs['yticks'])
        if 'labels' in kwargs:
            ax.legend(kwargs['labels'], fontsize=fontsize)
        if 'title' in kwargs:
            ax.set_title(kwargs['title'], fontsize=fontsize)
        ax.set_xscale(kwargs.get('xscale', 'linear'))
        ax.set_yscale(kwargs.get('yscale', 'linear'))
        return ax

    def plot_lines(self, x, y, ax=None, colormap=None, **kwargs):
        x = np.asarray(x)
        y = np.asarray(y)
        ny = len(y)

        if ax is None:
            fig, ax = plt.subplots()

        if x.ndim != y.ndim:
            if x.ndim in (0, 1):
                x = np.tile(x, (ny, 1))
            else:
                raise ValueError('Outer dimension of x is not one and not '
                                 'equal to outer dimension of y')
        if y.ndim == 1:
            ax.plot(x, y, marker=kwargs.get('marker', '.'),
                          markersize=kwargs.get('markersize', MARKER_SIZE),
                          fillstyle=kwargs.get('fillstyle', 'full'),
                          linewidth=kwargs.get('linewidth', LINE_WIDTH),
                          linestyle=kwargs.get('linestyle', '-'),
                          color=kwargs.get('color', 'k'))
        else:
            if colormap is not None:
                cmap = plt.get_cmap(colormap)
                colors = cmap(np.linspace(0.0, 1.0, ny))
            else:
                colors = \
                    kwargs.get('color',
                               list(islice(cycle(['k', 'b', 'r', 'g', 'y']),
                                           ny)))
            linestyles = \
                list(islice(cycle(kwargs.get('linestyle', ['-'])), ny))
            markers = \
                list(islice(cycle(kwargs.get('marker', ['.'])), ny))
            fillstyles = \
                list(islice(cycle(kwargs.get('fillstyle', ['full'])), ny))
            for i in range(ny):
                ax.plot(x[i], y[i], marker=markers[i],
                        markersize=kwargs.get('markersize', MARKER_SIZE),
                        fillstyle=fillstyles[i],
                        linewidth=kwargs.get('linewidth', LINE_WIDTH),
                        linestyle=linestyles[i],
                        color=colors[i])
        set_props = kwargs.get('set_props', False)
        if set_props:
            ax.grid(True)
            ax.use_sticky_edges = False
            ax.autoscale()
            ax = self.set_ax_properties(ax, **kwargs)
        return ax

    def create_figure(self, filepath, x_array, y_array, xlabels, ylabels,
                      xlims=None, ylims=None, xticks=None, yticks=None,
                      titles=None, rows=1, cols=1, **kwargs):
        nplots = rows*cols

        def check_dims(variable, correct_single_dim=False):
            if isinstance(variable, str):
                variable = [variable]
            if not isinstance(variable, (list, tuple, np.ndarray)):
                raise TypeError('variable must be provided '
                                'as tuple, list or numpy array')
            if len(variable) != nplots:
                if correct_single_dim:
                    if nplots == 1:
                        variable = [variable]
                    else:
                        raise ValueError('variable must be sequence with '
                                         'length equivalent to number of plots')
                else:
                    raise ValueError('variable must be sequence with '
                                     'length equivalent to number of plots')
            return variable

        if rows > 2:
            figsize = kwargs.get('figsize', (FIG_SIZE[0],
                                             FIG_SIZE[1] * float(rows) / 2.0))
        else:
            figsize = kwargs.get('figsize', FIG_SIZE)
        fig = plt.figure(dpi=kwargs.get('dpi', FIG_DPI), figsize=figsize)

        x_array = np.asarray(x_array)
        y_array = check_dims(np.asarray(y_array), correct_single_dim=True)

        if len(x_array) != nplots:
            if x_array.ndim == 1:
                x_array = np.tile(x_array, (nplots, 1))
            else:
                raise ValueError('Dimension of x-array is not one and does not '
                                 'match number of plot')
        fontsize = kwargs.get('fontsize', FONT_SIZE)
        xlabels = check_dims(xlabels)
        ylabels = check_dims(ylabels)

        for i in range(nplots):
            ax = fig.add_subplot(rows, cols, i+1)
            xarray = x_array[i]
            yarray = y_array[i]
            ax = self.plot_lines(x_array[i], y_array[i], ax=ax,
                                 xlabel=xlabels[i], ylabel=ylabels[i], **kwargs)
            if titles is not None:
                titles = check_dims(titles)
                ax.set_title(titles[i], fontsize=fontsize)
            if 'legend' in kwargs:
                legend = check_dims(kwargs['legend'], correct_single_dim=True)
                ax.legend(legend[i])
            if xlims is not None:
                xlims = check_dims(xlims, correct_single_dim=True)
                ax.set_xlim(xlims[i])
            if ylims is not None:
                xlims = check_dims(xlims, correct_single_dim=True)
                ax.set_ylim(ylims[i])
            if xticks is not None:
                xticks = check_dims(xticks, correct_single_dim=True)
                ax.set_xticks(xticks[i])
            if yticks is not None:
                xlims = check_dims(yticks, correct_single_dim=True)
                ax.set_yticks(yticks[i])
        plt.tight_layout()
        if filepath:
            fig.savefig(filepath, format=kwargs.get('fileformat', 'png'))
        return fig

    def write_array_to_csv(self, file_path, array, header=None,
                           separator_lines=None, mode='a'):
        with open(file_path, mode) as file:
            if header is not None:
                file.write('# ' + header + '\n')
            if separator_lines is not None:
                for i in range(len(separator_lines)):
                    a = array[i]
                    if a.ndim == 1:
                        a = a.reshape(1, a.shape[0])
                    file.write(separator_lines[i])
                    np.savetxt(file, a,
                               delimiter=self.delimiter, fmt=self.csv_format)
            else:
                np.savetxt(file, array,
                           delimiter=self.delimiter, fmt=self.csv_format)
            return file
