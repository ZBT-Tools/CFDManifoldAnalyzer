import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cfd_manifold_analyzer.settings.file_names as path

"""
Read AVL 2D data, extract mass flows and save as binary numpy data
"""

full_output_dir = os.path.join(path.dir_name, path.output_dir)

# create output folder
if not os.path.isdir(full_output_dir):
    os.makedirs(full_output_dir)

avl_data = \
    pd.read_csv(os.path.join(path.dir_name, path.avl_fire_file_2d),
                sep='\t', header=[0, 1])

channel_mass_flows = avl_data[path.mass_flow_name].iloc[-2].to_numpy()
channel_numbers = range(len(channel_mass_flows))
total_mass_flow = avl_data[path.total_mass_flow_name].iloc[-2][0]

mean_mass_flow = channel_mass_flows.mean()
plt.plot(channel_numbers, channel_mass_flows / mean_mass_flow)

np.save(os.path.join(full_output_dir, path.mass_flow_data_file),
        channel_mass_flows)
