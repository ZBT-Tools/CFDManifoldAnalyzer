"""
Specify manifold geometry case simulated with AVL FIRE:
all data in y-z-coordinates according to setup in AVL FIRE while y-direction is
along channels and z-direction is along manifolds
"""

import numpy as np

# number of parallel channels
n_channels = 16
# number of manifolds
n_manifolds = 2
# manifold centerline height [m] (y-coordinate),
# tuple indexing along y-coordinate not according to flow direction
manifold_y = (0, 0.24)
# flow direction in manifold in z-direction,
# tuple indexing along y-coordinate not according to flow direction
manifold_flow_direction_z = (-1, 1)
# flow direction in channel in y-direction
channel_flow_direction_y = -1
# discretization [m] in channel along y-axis
channel_dy = 5e-4
# discretization [m] in manifold along z-direction
manifold_dz = 5e-4
# distance between manifolds in y-direction [m]
manifold_distance = manifold_y[1] - manifold_y[0]
# manifold diameter [m]
manifold_diameter = 0.01
# channel diameter [m]
channel_diameter = 0.005
channel_length = manifold_distance - manifold_diameter
# distance between adjacent channels in z-direction [m]
channel_distance_z = 0.02
# z-coordinate of first channel [m]
channel_0_z = 0.005
# maximum extensions of coordinates in 3D results cut [m]
x_ext = (0.0, 0.0)
y_ext = (0.0 - 0.5 * manifold_diameter,
         manifold_distance + 0.5 * manifold_diameter)
z_ext = (-0.105, 0.315)
bounding_box = np.asarray((x_ext, y_ext, z_ext))
