"""
Paths to AVL FIRE output data: directory, 3D cut file, 2D results file
- directory: local directory containing result files
- 3D cut file: coordinates and pressures in cut
- 2D results file: mass flows for each channel
"""

# AVL FIRE results directory
dir_name = r'C:\Users\feierabend\PycharmProjects\CFDManifoldAnalyzer\data'

# 3D cut file name
# Each row of file has three vertices with their respective pressure values:
# (without header and values separated by single whitespace character)
# x1 y1 z1 x2 y2 z2 x3 y3 z3 p1 p2 p3
# avl_fire_file_3d = \
#     'Plane_Cut_0002_Stack_Ratio2_Re3000_p05_Long_IT_2517_Flow_RelativePressure_Pa.dat'
avl_fire_file_3d = \
    r'FIRE_M\FIRE_M_2.split_variation.Case_1\results\values_in_All_at_1124_proc_1.dat'
# 2D results file name
# Each data row includes all 2D results for the respective iteration; first two
# rows are headers with result name and dimension in first and second row,
# respectively. Mass flows for 2D cell selections are usually named "Flow:
# Mass Flow", however if different please specify name below
avl_fire_file_2d = 'GuiGraph_02-Case_Stack_Ratio2_Re3000_p05_Long_Fire.txt'
mass_flow_name = 'Flow:Mass Flow'
total_mass_flow_name = 'Flow:Total Inlet Massflow'

# Output names
output_dir = 'ReChl-40'
output_main_name = 'combining_t-junction'
mass_flow_data_file = 'mass_flows'
channel_data_file = 'channel_data'
manifold_data_file = 'manifold_data'
