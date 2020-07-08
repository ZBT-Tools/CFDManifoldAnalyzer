import os
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import sys
sys.path.append(os.path.abspath('../../'))
import PEMFCModel.channel as chl

class Channels:
    """
    Class to process AVL FIRE simulation channel data
    """
    def __init__(self):
        self.number = geom.n_channels
        self.flow_direction = (0.0, -1.0, 0.0)
        self.dy = geom.channel_dy
        self.distance = geom.channel_distance_z
        self.diameter = geom.channel_diameter
        self.initial_z = geom.channel_0_z
        self.data_name = files.channel_data_file

    # def load_data(self):



