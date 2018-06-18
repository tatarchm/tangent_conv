import sys

open3d_path = '/misc/lmbraid17/tatarchm/projects/tangent_conv/dev/Open3D/build/lib/'
tc_path = '/misc/lmbraid17/tatarchm/projects/tangent_conv/dev/tangent_convolutions/'

sys.path.append(open3d_path)
from py3d import *

def get_tc_path():
	return tc_path