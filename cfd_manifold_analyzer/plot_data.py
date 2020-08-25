# global imports
import os
import matplotlib.pyplot as plt
import numpy as np

# local imports
from cfd_manifold_analyzer.settings import file_names

output_dir = os.path.join(file_names.dir_name, file_names.output_dir)
if 'xy_array' not in globals():
    xy_array = np.loadtxt(os.path.join(output_dir,
                                       file_names.output_main_name + '.txt'))
labels = ['$\zeta$-Manifold-Manifold', '$\zeta$-Manifold-Branch',
          None, None]
xlabels = ['Discharge Ratio [-]', 'Discharge Ratio [-]', 'Discharge Ratio [-]',
           'Discharge Ratio [-]']
ylabels = ['Resistance Coefficient [-]', 'Resistance Coefficient [-]',
           'Manifold Reynolds Number [-]', 'Manifold Reynolds Number [-]']
file_names_ext = \
    ['_resistance_mfd_mfd.png', '_resistance_mfd_chl.png',
     '_manifold_reynolds_number.png', '_channel_reynolds_number.png']
#yscales = ['linear', 'linear', 'linear', 'linear']
yscales = ['linear', 'symlog', 'symlog', 'linear']


for i in range(len(xy_array) - 1):
    label = labels[i]
    fig, ax = plt.subplots()
    ax.plot(xy_array[0], xy_array[i + 1], label=label, marker='.')
    if label is not None:
        ax.legend()
    ax.set_yscale(yscales[i])
    ax.set_xticks(np.arange(0.0, 1.0, 0.1))
    ax.set_xlabel(xlabels[i])
    ax.set_ylabel(ylabels[i])
    ax.grid()
    fig.savefig(os.path.join(output_dir,
                             file_names.output_main_name + file_names_ext[i]))
