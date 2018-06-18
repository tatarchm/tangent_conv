import numpy as np
import tensorflow as tf
import sys
import os
import random

from path_config import *
from cloud import *
from util import *

class param:
	def __init__(self, config):
		self.experiment_dir = os.path.join(get_tc_path(), config['co_experiment_dir'])
		self.output_dir = os.path.join(self.experiment_dir, config['co_output_dir'])
		self.test_file = os.path.join(get_tc_path(), config['co_test_file'])
		self.dataset_dir = os.path.join(get_tc_path(), config['pre_dataset_dir'])
		self.scan_file = config['eval_scan_file']
		self.label_file = config['eval_label_file']
		self.output_file = config['eval_output_file']

		self.min_cube_size = config['pre_min_cube_size']
		self.cube_size = [self.min_cube_size, 2*self.min_cube_size, 4*self.min_cube_size]
		self.num_scales = len(self.cube_size)

def run_extrapolate_labels(config):
	avg_acc = 0.0
	p = param(config)
	with open(p.test_file) as f:
		scans = f.readlines()
		scans = [s.rstrip() for s in scans]
		for scan_name in scans:
			s = ScanData()
			s.load(os.path.join(p.experiment_dir, p.output_dir, scan_name), p.num_scales)

			full_scan_path = os.path.join(p.dataset_dir, scan_name)
			ref_cloud = read_point_cloud(os.path.join(full_scan_path, p.scan_file))
			ref_labels = read_txt_labels(os.path.join(full_scan_path, p.label_file))

			l, a = s.extrapolate_labels(ref_cloud, ref_labels)
			avg_acc += a
			print("oA " + scan_name + ": " + str(a))
			out_file_name = os.path.join(p.output_dir, scan_name, p.output_file)
			with open(out_file_name, "w") as f:
				f.writelines(["%s\n" % item  for item in l])

	return 0
