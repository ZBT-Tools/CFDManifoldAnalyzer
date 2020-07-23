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
# manifold centerline y-coordinate [m]
# tuple indexing according to flow direction
# (index 0: inlet manifold, index 1: outlet manifold)
manifold_y = (0.24, 0.0)
# flow direction in manifold in z-direction,
# tuple indexing according to flow direction
# (index 0: inlet manifold, index 1: outlet manifold)
manifold_flow_direction = ((0.0, 0.0, 1.0),
                           (0.0, 0.0, -1.0))
# flow direction in channel in y-direction
channel_flow_direction = (0.0, -1.0, 0.0)
# discretization [m] in channel along y-axis
channel_dy = 5e-4
# discretization [m] in manifold along z-direction
manifold_dz = 5e-4
# distance between manifolds in y-direction [m]
manifold_distance = np.abs(manifold_y[1] - manifold_y[0])
# manifold diameter [m]
manifold_diameter = 0.01
# channel diameter [m]
channel_diameter = 0.005
channel_length = manifold_distance - manifold_diameter
# distance between adjacent channels in z-direction [m]
channel_distance_z = 0.02
# z-coordinate of first channel [m]
channel_0_z = 0.005
# linear segments of channel pressure distribution
lin_segments = ((0.011, 0.016), (0.221, 0.226))
# maximum extensions of coordinates in 3D results cut [m]
x_ext = (0.0, 0.0)
y_ext = (0.0 - 0.5 * manifold_diameter,
         manifold_distance + 0.5 * manifold_diameter)
z_ext = (-0.105, 0.315)
bounding_box = np.asarray((x_ext, y_ext, z_ext))
manifold_length = np.abs(bounding_box[-1, 1] - bounding_box[-1, 0])

# coordinate range for manifold pressure analyzation
manifold_range = (-0.01, 0.3)


