import cfd_manifold_analyzer.settings.file_names as path
import numpy as np
import os

"""
Convert AVL FIRE 3D cut ascii data to numpy binary file
"""
output_name = path.avl_fire_file_3d.split('.')[0]
data = np.loadtxt(os.path.join(path.dir_name, path.avl_fire_file_3d))
np.save(os.path.join(path.dir_name, output_name), data)
