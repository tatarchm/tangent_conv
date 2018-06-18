import numpy as np
import os

from joblib import Parallel, delayed
import multiprocessing
import subprocess

from util.path_config import *

def read_txt_labels(file_path):
	with open(file_path) as f:
		labels = f.readlines()
		lb = [c.rstrip() for c in labels]
	return np.asarray(lb, dtype='int32')

def remap_colors(lb, color_map):
	out = []
	for l in lb:
		out.append(color_map[int(l)])
	return np.asarray(out)

def expand_dim_to_batch2(array, local_batch_size, dummy_val=-1):
	sp = np.shape(array)
	out_arr = np.zeros((local_batch_size, sp[1]))
	out_arr.fill(dummy_val)
	out_arr[0:sp[0], :] = array
	return out_arr

def expand_dim_to_batch1(array, batch_size, dummy_val=0):
	sp = np.shape(array)
	out_arr = np.zeros((batch_size))
	out_arr.fill(dummy_val)
	out_arr[0:sp[0]] = array
	return out_arr

def get_pooling_mask(pooling):
	mask = np.asarray(pooling > 0, dtype='float32')
	sp = np.shape(mask)
	for i in range(0, sp[0]):
		mult = np.count_nonzero(mask[i, :])
		if mult == 0:
			mask[i, :] *= 0
		else:
			mask[i, :] *= 1/mult
	return mask

def invert_index_map(idx_map):
	remapped_idx = {}
	for i in range(0, len(idx_map)):
		remapped_idx[idx_map[i]] = i
	return remapped_idx

def remap_indices(idx_map, cloud_subset):
	sp = np.shape(cloud_subset)
	cloud_subset.flatten()
	out = [idx_map.get(i, -1) for i in cloud_subset.flatten()]
	return np.reshape(np.asarray(out), (sp[0], sp[1]))

class ScanData():

	def __init__(self):
		self.clouds = []
		self.labels_gt = []
		self.labels_pr = []
		self.conv_ind = []
		self.pool_ind = []
		self.depth = []
		self.trees = []

	def load(self, file_path, num_scales):
		for i in range(0, num_scales):
			fname = os.path.join(file_path, "scale_" + str(i) + ".npz")
			l = np.load(fname)

			cloud = PointCloud()
			cloud.points = Vector3dVector(l['points'])
			cloud.colors = Vector3dVector(l['colors'])
			tree = KDTreeFlann(cloud)

			estimate_normals(cloud, search_param=KDTreeSearchParamHybrid(radius=0.5, max_nn=100))

			self.clouds.append(cloud)
			self.labels_gt.append(l['labels_gt'])
			self.conv_ind.append(l['nn_conv_ind'])
			self.depth.append(l['depth'])
			self.trees.append(tree)
			self.pool_ind.append(l['pool_ind'])
			if 'labels_pr' in l.keys():
				self.labels_pr.append(l['labels_pr'])
			else:
				self.labels_pr.append(np.zeros(np.shape(self.labels_gt[i])))
		print("loaded " + file_path.split('/')[-2])

	def save(self, file_path, num_scales=3):
		for i in range(0, num_scales):
			np.savez_compressed(os.path.join(file_path, 'scale_' + str(i) + '.npz'),
					points=np.asarray(self.clouds[i].points),
					colors=np.asarray(self.clouds[i].colors),
					labels_gt=self.labels_gt[i],
					labels_pr=self.labels_pr[i],
					nn_conv_ind=self.conv_ind[i],
					pool_ind=self.pool_ind[i],
					depth=self.depth[i])
				
	def remap_depth(self, vmin, vmax):
		num_scales = len(self.depth)
		for i in range(0, num_scales):
			self.depth[i] = np.clip(self.depth[i], vmin, vmax)
			self.depth[i] -= vmin
			self.depth[i] *= 1.0 / (vmax - vmin)

	def has_points(self, point, radius):
		[k, idx_valid, _] = self.trees[0].search_radius_vector_3d(point, radius=radius)
		return np.count_nonzero(self.labels_gt[0][idx_valid]) > 0

	def get_random_valid_point(self, radius=None):
		h = False
		while not h:
			num_points = np.shape(np.asarray(self.clouds[0].points))[0]
			random_ind = random.randint(0, num_points-1)
			random_point = np.asarray(self.clouds[0].points)[random_ind, :]
			if radius is not None:
				h = self.has_points(random_point, radius)
			else:
				h = self.labels_gt[0][random_ind] > 0
		return random_point, random_ind

	def remap_normals(self, vmin=-1.0, vmax=1.0):
		num_scales = len(self.clouds)
		for i in range(0, num_scales):
			normals = np.asarray(self.clouds[i].normals)
			normals = np.clip(normals, vmin, vmax)
			normals -= vmin
			normals *= 1.0 / (vmax - vmin)
			self.clouds[i].normals = Vector3dVector(normals)

	def get_height(self, scale=0):
		raw_z = np.asarray(self.clouds[scale].points)[:, 2:3]
		max_z = np.max(raw_z)
		min_z = np.min(raw_z)
		return (raw_z - min_z) * 1.0 / (max_z - min_z)

	def assign_labels(self, pr_arr):
		for i in range(0, len(pr_arr)):
			if pr_arr[i] > 0:
				self.labels_pr[0][i] = pr_arr[i]
		self.labels_pr[1] = None
		self.labels_pr[2] = None

	def assign_labels_part(self, pr_arr, idx_map):
		for i in range(0, len(idx_map)):
			if pr_arr[i] > 0:
				self.labels_pr[0][idx_map[i]] = pr_arr[i]
		self.labels_pr[1] = None
		self.labels_pr[2] = None

	def extrapolate_labels(self, ref_cloud, ref_labels):
		out_labels = []
		points = np.asarray(ref_cloud.points)
		cnt = 0
		total_lb = 0
		correct_lb = 0

		idx_val = self.labels_pr[0] > 0

		labeled_cloud = PointCloud()
		labeled_cloud.points = Vector3dVector(np.asarray(self.clouds[0].points)[idx_val, :])
		labeled_cloud.colors = Vector3dVector(np.asarray(self.clouds[0].colors)[idx_val, :])
		search_tree = KDTreeFlann(labeled_cloud)
		valid_labels = self.labels_pr[0][idx_val]

		for pt in points:
			gt_lb = ref_labels[cnt]
			if gt_lb > 0:
				[k, idx, _] = search_tree.search_knn_vector_3d(pt, 1)
				pr_lb = valid_labels[idx]
				out_labels.append(int(pr_lb))
				if pr_lb == gt_lb:
					correct_lb += 1
				total_lb += 1
			else:
				out_labels.append(int(0))
			cnt += 1
		accuracy = correct_lb / total_lb
		return out_labels, accuracy


class BatchData():

	def __init__(self):
		self.colors = []
		self.normals = []
		self.conv_ind = []
		self.pool_ind = []
		self.labels = []
		self.depth = []
		self.index_maps = []
		self.loss_weights = []
		self.height = []

	def num_points(self):
		return np.shape(self.colors[0])[0]

def get_batch_from_full_scan(full_scan, num_scales, class_weights):
	out = BatchData()
	for i in range(0, num_scales):
		out.colors.append(np.asarray(full_scan.clouds[i].colors))
		out.normals.append(np.asarray(full_scan.clouds[i].normals))
		out.depth.append(np.asarray(full_scan.depth[i]))
		out.labels.append(np.asarray(full_scan.labels_gt[i]))
		out.conv_ind.append(full_scan.conv_ind[i].T)
		if i > 0:
			p_ind = full_scan.pool_ind[i]
		else:
			p_ind = None
		out.pool_ind.append(p_ind)

		curr_w = []
		for lb in full_scan.labels_gt[i]:
			if lb > 0:
				curr_w.append(class_weights[int(lb-1)])
			else:
				curr_w.append(0.0)
		out.loss_weights.append(curr_w)
		out.height.append(full_scan.get_height(i))
	return out

def get_scan_part_out(par, point=None, sample_type='POINT'):
	lb_count = 0
	num_scales = len(scan.clouds)
	while lb_count == 0:
		if point is None:
			if sample_type == 'SPACE':
				min_bound = scan.clouds[0].get_min_bound()
				max_bound = scan.clouds[0].get_max_bound()
				ext = max_bound - min_bound
				random_point = np.random.rand(3) * ext + min_bound
			elif sample_type == 'POINT':
				num_points = np.shape(np.asarray(scan.clouds[0].points))[0]
				random_ind = random.randint(0, num_points)
				random_point = np.asarray(scan.clouds[0].points)[random_ind, :]
		else:
			random_point = np.asarray(point)

		[k_valid, idx_valid, _] = scan.trees[0].search_radius_vector_3d(random_point, radius=par.valid_rad)
		idx_valid = np.asarray(idx_valid)
		lbl = np.asarray(scan.labels_gt[0][idx_valid])
		lb_count = np.count_nonzero(lbl)

		if (point is not None) and (lb_count == 0):
			return None

	idx_maps = []
	out = BatchData()

	for i in range(0, num_scales):
		[k, idx, _] = scan.trees[i].search_radius_vector_3d(random_point, radius=par.full_rf_size())

		out.colors.append(np.asarray(scan.clouds[i].colors)[idx, :])
		out.normals.append(np.asarray(scan.clouds[i].normals)[idx, :])
		out.depth.append(np.asarray(scan.depth[i])[:, idx])
		out.labels.append(np.asarray(scan.labels_gt[i][idx]))
		out.height.append(scan.get_height(i)[idx, :])

		curr_w = []
		for lb in scan.labels_gt[i][idx]:
			if lb > 0:
				curr_w.append(par.d_par.class_weights[int(lb-1)])
			else:
				curr_w.append(0.0)
		out.loss_weights.append(curr_w)

		valid_points = []
		if i == 0:
			cnt = 0
			for l in idx:
				if l not in idx_valid:
					out.labels[0][cnt] = 0
				else:
					valid_points.append(cnt)
				cnt += 1

		idx_map = invert_index_map(idx)
		idx_maps.append(idx_map)
		out.index_maps.append(np.asarray(idx))

	valid_labeled = np.count_nonzero(np.asarray(out.labels[0]))
	# print("Valid labeled: " + str(valid_labeled))

	for i in range(0, num_scales):
		ci = scan.conv_ind[i][:, out.index_maps[i]].T
		ci = remap_indices(idx_maps[i], ci)
		out.conv_ind.append(ci)
		# print(conv_ind)

		if i > 0:
			pool_ind = scan.pool_ind[i][out.index_maps[i], :]
			pool_ind = remap_indices(idx_maps[i-1], pool_ind)
		else:
			pool_ind = None

		out.pool_ind.append(pool_ind)

	return out

def get_batch_array(scan_var, par, points=None):
	global scan
	scan = scan_var
	num_cores = multiprocessing.cpu_count()

	if points is None:
		pts = []
		for i in range(0, par.batch_array_size):
			pt, ind = scan_var.get_random_valid_point(par.valid_rad)
			pts.append(pt)
		arr_size = par.batch_array_size
	else:
		pts = points
		arr_size = len(points)
		print(arr_size)

	batch_array = Parallel(n_jobs=num_cores)(
			delayed(get_scan_part_out)(par, pts[i]) for i in range(0, arr_size))
	return batch_array