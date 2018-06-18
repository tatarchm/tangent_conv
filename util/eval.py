import numpy as np
import tensorflow as tf
import sys
import os
import random

from path_config import *
from cloud import *
from dataset_params import *

class param:
	def __init__(self, config):
		self.experiment_dir = os.path.join(get_tc_path(), config['co_experiment_dir'])
		self.output_dir = os.path.join(self.experiment_dir, config['co_output_dir'])
		self.test_file = os.path.join(get_tc_path(), config['co_test_file'])
		self.dataset_dir = os.path.join(get_tc_path(), config['pre_dataset_dir'])
		self.label_file = config["eval_label_file"]
		self.output_file = config["eval_output_file"]
		dataset_type = config['pre_dataset_param']
		if dataset_type == "stanford":
			self.d_par = stanford_params()
		elif dataset_type == "scannet":
			self.d_par = scannet_params()
		elif dataset_type == "semantic3d":
			self.d_par = semantic3d_params()

def build_conf_matrix(gt_list, pr_list):
	cnt = 0
	global conf_mat, classes
	for gt_l in gt_list:
		if gt_l > 0:
			pr_l = pr_list[cnt]
			conf_mat[gt_l-1, pr_l-1] += 1
		cnt += 1

def get_iou():
	out = []
	avg = 0.0
	global conf_mat, classes
	for cl in classes:
		nom = conf_mat[cl-1, cl-1]
		denom = sum(conf_mat[cl-1, :]) + sum(conf_mat[:, cl-1]) - conf_mat[cl-1, cl-1]
		if denom > 0:
			out.append(nom / denom)
		else:
			out.append(0.0)
		avg += out[cl-1]
	print(out)
	print("mIoU: " + str(avg / len(classes)))
	return out

def get_o_acc():
	s_corr = 0.0
	global conf_mat, classes
	for i in range(0, len(classes)):
		s_corr += conf_mat[i, i]
	oa = s_corr / np.sum(conf_mat)
	print("oA: " + str(oa))
	return oa

def get_acc():
	out = []
	avg = 0.0
	global conf_mat, classes
	for cl in classes:
		nom = conf_mat[cl-1, cl-1]
		denom = sum(conf_mat[cl-1, :])
		if denom > 0:
			out.append(nom / denom)
		else:
			out.append(0.0)
		avg += out[cl-1]
	print(out)
	print("mA: " + str(avg / len(classes)))
	return out


def run_eval(config):

	p = param(config)
	global conf_mat, classes
	conf_mat = np.zeros((p.d_par.num_classes, p.d_par.num_classes))
	classes = list(range(1, p.d_par.num_classes))

	# exclude class 'stairs' from evaluation for S3DIS
	if isinstance(p.d_par, stanford_params):
		classes = classes[:-1]

	with open(p.test_file) as f:
		scans = f.readlines()
		scans = [s.rstrip() for s in scans]

		counter = 0

		avg_iou = []
		cnt_iou = []
		avg_acc = []
		cnt_acc = []

		for i in range(0, len(classes)):
			avg_iou.append(0.0)
			cnt_iou.append(0)
			avg_acc.append(0.0)
			cnt_acc.append(0)

		cnt = 0
		for scan_name in scans:
			print(scan_name)
			full_scan_path = os.path.join(p.dataset_dir, scan_name)
			ref_labels = read_txt_labels(os.path.join(full_scan_path, p.label_file))
			pr_scan_path = os.path.join(p.output_dir, scan_name, p.output_file)
			pr_labels = read_txt_labels(pr_scan_path)
			counter += 1
			build_conf_matrix(ref_labels, pr_labels)
			cnt += 1

		get_iou()
		get_acc()
		get_o_acc()

	return 0
