"""
Specify manifold geometry case simulated with AVL FIRE:
all data in y-z-coordinates according to setup in AVL FIRE while y-direction is
along channels and z-direction is along manifolds
"""

import numpy as np

# number of parallel channels
n_channels = 1
# number of manifolds
n_manifolds = 1
# manifold centerline y-coordinate [m]
# list indexed according to flow direction
# (index 0: inlet manifold, index 1: outlet manifold)
manifold_y = [0.0]
# flow direction in manifold in z-direction,
# list indexed according to flow direction
# (index 0: inlet manifold, index 1: outlet manifold)
manifold_flow_direction = [[0.0, 0.0, 1.0]]
# flow direction in channel in y-direction
channel_flow_direction = [0.0, 1.0, 0.0]
# discretization [m] in channel along y-axis
channel_dy = 1e-3
# discretization [m] in manifold along z-direction
manifold_dz = 1e-3
# distance between manifolds in y-direction [m]
manifold_distance = 0.0  # np.abs(manifold_y[1] - manifold_y[0])
# manifold diameter [m]
manifold_diameter = 0.01
# channel diameter [m]
channel_diameter = 0.005
channel_length = 0.15
# distance between adjacent channels in z-direction [m]
channel_distance_z = 0.00
# z-coordinate of first channel [m]
channel_0_z = 0.0
# linear segments of channel pressure distribution
lin_segments = [[0.08, 0.13]]
# maximum extensions of coordinates in 3D results cut [m]
x_ext = [-0.005, 0.005]
y_ext = [-0.5 * manifold_diameter, channel_length + 0.5 * manifold_diameter]
z_ext = [-0.2, 0.2]
bounding_box = np.asarray((x_ext, y_ext, z_ext))
manifold_length = np.abs(bounding_box[-1, 1] - bounding_box[-1, 0])

# coordinate range for manifold pressure analyzation
manifold_range = (-0.1, 0.1)


