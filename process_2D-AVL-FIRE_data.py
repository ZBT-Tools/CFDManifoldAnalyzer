# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:14:26 2020

@author: feierabend
"""

import numpy as np
import scipy as sp
from scipy.interpolate import griddata
import pandas as pd
import os
import matplotlib.pyplot as plt
import timeit


start_time = timeit.default_timer()

# specify directory and file names
dir_name = r'D:\Projekte\FlowDistribution\Manifold_Modell'
file_name = \
    '2D_Stack_Ratio2_Re3000_p10_Fire.txt'
output_dir = 'Case_1'
full_output_dir = os.path.join(dir_name, output_dir)

# create output folder
if not os.path.isdir(full_output_dir):
    os.mkdir(full_output_dir)

avl_data = \
    pd.read_csv(os.path.join(dir_name, file_name), sep='\t', header=[0,1])

channel_mass_flows = avl_data['Flow:Mass Flow'].iloc[-2].to_numpy()
channel_numbers = range(len(channel_mass_flows))
total_mass_flow = avl_data['Flow:Total Inlet Massflow'].iloc[-2][0]

mean_mass_flow = channel_mass_flows.mean()
plt.plot(channel_numbers, channel_mass_flows / mean_mass_flow)

np.save(os.path.join(full_output_dir, 'mass_flows'), channel_mass_flows)
