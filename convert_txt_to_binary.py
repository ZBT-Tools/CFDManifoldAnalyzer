import numpy as np
import os

input_name = \
    'Stack_Ratio2_Re3000_p01_IT_1194_Flow_RelativePressure_Pa.dat'
dir_name = r'D:\ZBT\Projekte\Manifold_Modell'

output_name = input_name.split('.')[0]

data = np.loadtxt(os.path.join(dir_name, input_name))

np.save(os.path.join(dir_name, output_name), data)