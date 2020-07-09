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

from ..settings import geometry
from ..settings import file_names


class CFDDataFlowCircuit(pemfc.flow_circuit.ParallelFlowCircuit):
    def __init__(self, dict_flow_circuit, manifolds, channels,
                 n_subchannels=1.0, **kwargs):

        super().__init__(dict_flow_circuit, manifolds, channels,
                         n_subchannels, **kwargs)

    def single_loop(self, inlet_mass_flow=None, update_channels=True):
        pass

    def update(self, inlet_mass_flow=None, calc_distribution=None):
        pass
