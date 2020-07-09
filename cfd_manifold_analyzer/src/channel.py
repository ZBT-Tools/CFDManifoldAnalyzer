# general imports
import os
import sys
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pemfc

# local module imports
from ..settings import geometry
from ..settings import file_names


class CFDDataChannels(pemfc.channel.IncompressibleFluidChannel):
    """
    Class to process AVL FIRE simulation channel data
    """
    def __init__(self):
        self.number = geometry.n_channels
        self.flow_direction = (0.0, -1.0, 0.0)
        self.dy = geometry.channel_dy
        self.distance = geometry.channel_distance_z
        self.diameter = geometry.channel_diameter
        self.initial_z = geometry.channel_0_z
        self.data_name = file_names.channel_data_file

    # def load_data(self):



