import os
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from settings.file_names import *
from settings.geometry import *


class Manifold:
    """
    Class to process AVL FIRE simulation manifold data
    """
    def __init__(self):
        pass