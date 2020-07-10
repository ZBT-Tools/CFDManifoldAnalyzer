import sys
import os
import pemfc
# import cfd_manifold_analyzer.src.convert_to_binary
# import cfd_manifold_analyzer.src.process_3d_avl_fire_cut
from cfd_manifold_analyzer.settings import file_names
# from cfd_manifold_analyzer.settings import geometry
import cfd_manifold_analyzer.src.cfd_data_processor as cfd_proc

file_path = os.path.join(file_names.dir_name, file_names.avl_fire_file_3d)
output_dir = os.path.join(file_names.dir_name, file_names.output_dir)
cfd_data = cfd_proc.CFDManifoldProcessor3D(file_path, output_dir)

# channel = pemfc.channel.Channel()
