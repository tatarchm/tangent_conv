import sys

open3d_path = '<your_path>/Open3D/build/lib/'
tc_path = '<your_path>/tangent_convolutions/'

sys.path.append(open3d_path)
from py3d import *

def get_tc_path():
	return tc_path
