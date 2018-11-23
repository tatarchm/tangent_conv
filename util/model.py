import numpy as np
import tensorflow as tf
from time import gmtime, strftime
import random

from common import *
from cloud import *
from dataset_params import *
from point_ops import *
from general_ops import *

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

class param:
	def __init__(self, config):
		self.pre_output_dir = os.path.join(get_tc_path(), config['pre_output_dir'])
		self.experiment_dir = os.path.join(get_tc_path(), config['co_experiment_dir'])
		self.output_dir = os.path.join(self.experiment_dir, config['co_output_dir'])
		self.train_file = os.path.join(get_tc_path(), config['co_train_file'])
		self.test_file = os.path.join(get_tc_path(), config['co_test_file'])
		self.num_rotations = config['pre_num_rotations']
		self.log_dir = os.path.join(self.experiment_dir, config['tt_log_dir'])
		self.snapshot_dir = os.path.join(self.experiment_dir, config['tt_snapshot_dir'])
		dataset_type = config['pre_dataset_param']
		if dataset_type == "stanford":
			self.d_par = stanford_params()
		elif dataset_type == "scannet":
			self.d_par = scannet_params()
		elif dataset_type == "semantic3d":
			self.d_par = semantic3d_params()
		self.input_type = config['tt_input_type']
		self.max_snapshots = config['tt_max_snapshots']
		self.test_iter = config['tt_test_iter']
		self.reload_iter = config['tt_reload_iter']
		self.max_iter_count = config['tt_max_iter_count']
		self.batch_size = config['tt_batch_size']
		self.valid_rad = config['tt_valid_rad']
		self.filter_size = config['tt_filter_size']
		self.batch_array_size = config['tt_batch_array_size']

		self.min_cube_size = config['pre_min_cube_size']
		self.cube_size = [self.min_cube_size, 2*self.min_cube_size, 4*self.min_cube_size]
		self.conv_rad = 2 * np.asarray(self.cube_size)
		self.num_scales = len(self.cube_size)

		###

		if isinstance(self.d_par, semantic3d_params):
			self.data_sampling_type = 'part'
		else:
			self.data_sampling_type = 'full'

	def full_rf_size(self):
		return self.valid_rad + 4*self.conv_rad[0] + 4*self.conv_rad[1] + 2*self.conv_rad[2]


class model():

	def __init__(self, curr_param):
		self.sess = tf.Session()
		self.training_step = 0
		self.par = curr_param

	def load_data(self, mode):
		if mode == "train":
			file_name = self.par.train_file
			self.training_data = []
		else:
			file_name = self.par.test_file
			self.test_data = []

		with open(file_name) as f:
			scans = f.readlines()

		scans = [s.rstrip() for s in scans]

		if mode == "train":
			self.training_scans = scans
		else:
			self.test_scans = scans

		scans = [os.path.join(self.par.pre_output_dir, s.rstrip()) for s in scans]

		cnt = 0
		for s_path in scans:
			if mode == "train":
				rot = random.randint(0, self.par.num_rotations-1)
			else:
				rot = 0
			s = ScanData()
			s.load(os.path.join(s_path, str(rot)), self.par.num_scales)
			s.remap_depth(vmin=-self.par.conv_rad[0], vmax=self.par.conv_rad[0])
			s.remap_normals()
			if mode == "train":
				self.training_data.append(s)
			else:
				self.test_data.append(s)
			cnt += 1

	def precompute_validation_batches(self):
		self.validation_batches = []
		for test_scan in self.test_data:
			if self.par.data_sampling_type == 'part':
				batch_array = get_batch_array(test_scan, self.par)
				for b in batch_array:
					if np.shape(b.colors[0])[0] <= self.par.batch_size:
						self.validation_batches.append(b)
			else:
				b = get_batch_from_full_scan(test_scan, self.par.num_scales, self.par.d_par.class_weights)
				if np.shape(b.colors[0])[0] <= self.par.batch_size:
					self.validation_batches.append(b)

	def get_training_batch(self, iter_num):
		if self.par.data_sampling_type == 'full':
			num_train_scans = len(self.training_data)
			scan_num = iter_num % num_train_scans
			return get_batch_from_full_scan(self.training_data[scan_num], self.par.num_scales, self.par.d_par.class_weights)
		else:
			scan_num = iter_num % self.par.batch_array_size
			if scan_num == 0:
				random_scan = random.randint(0, len(self.training_data)-1)
				self.tr_batch_array = get_batch_array(self.training_data[random_scan], self.par)
			return self.tr_batch_array[scan_num]

	def get_feed_dict(self, b):
		bs = self.par.batch_size

		mask1 = get_pooling_mask(b.pool_ind[1])
		mask2 = get_pooling_mask(b.pool_ind[2])

		ret_dict = {self.c1_ind: expand_dim_to_batch2(b.conv_ind[0], bs),
					self.c2_ind: expand_dim_to_batch2(b.conv_ind[1], bs//2),
					self.c3_ind: expand_dim_to_batch2(b.conv_ind[2], bs//4),
					self.p12_ind: expand_dim_to_batch2(b.pool_ind[1], bs//2),
					self.p12_mask: expand_dim_to_batch2(mask1, bs//2, dummy_val=0),
					self.p23_ind: expand_dim_to_batch2(b.pool_ind[2], bs//4),
					self.p23_mask: expand_dim_to_batch2(mask2, bs//4, dummy_val=0),
					self.label: expand_dim_to_batch1(b.labels[0], bs),
					self.loss_weight: expand_dim_to_batch1(b.loss_weights[0], bs)}

		if 'd' in self.par.input_type:
			ret_dict.update({self.input_depth1: expand_dim_to_batch2(b.depth[0].T, bs)})
			ret_dict.update({self.input_depth2: expand_dim_to_batch2(b.depth[1].T, bs//2)})
			ret_dict.update({self.input_depth3: expand_dim_to_batch2(b.depth[2].T, bs//4)})
		if 'n' in self.par.input_type:
			ret_dict.update({self.input_normals: expand_dim_to_batch2(b.normals[0], bs)})
		if 'h' in self.par.input_type:
			ret_dict.update({self.input_h: expand_dim_to_batch2(b.height[0], bs)})
		if 'c' in self.par.input_type:
			ret_dict.update({self.input_colors: expand_dim_to_batch2(b.colors[0], bs)})

		return ret_dict

	def build_model(self, batch_size):
		self.best_accuracy = 0.0
		fs = self.par.filter_size
		bs = batch_size

		num_input_ch = 0
		input_list = []
		if 'd' in self.par.input_type:
			num_input_ch += 1
			self.input_depth1 = tf.placeholder(tf.float32, [bs, fs*fs])
			self.input_depth2 = tf.placeholder(tf.float32, [bs//2, fs*fs])
			self.input_depth3 = tf.placeholder(tf.float32, [bs//4, fs*fs])
		if 'n' in self.par.input_type:
			num_input_ch += 3
			self.input_normals = tf.placeholder(tf.float32, [bs, 3])
			input_list.append(self.input_normals)
		if 'h' in self.par.input_type:
			num_input_ch += 1
			self.input_h = tf.placeholder(tf.float32, [bs, 1])
			input_list.append(self.input_h)
		if 'c' in self.par.input_type:
			num_input_ch += 3
			self.input_colors = tf.placeholder(tf.float32, [bs, 3])
			input_list.append(self.input_colors)

		self.c1_ind = tf.placeholder(tf.int32, [bs, fs*fs])
		self.p12_ind = tf.placeholder(tf.int32, [bs//2, 8])
		self.p12_mask = tf.placeholder(tf.float32, [bs//2, 8])
		self.c2_ind = tf.placeholder(tf.int32, [bs//2, fs*fs])
		self.p23_ind = tf.placeholder(tf.int32, [bs//4, 8])
		self.p23_mask = tf.placeholder(tf.float32, [bs//4, 8])
		self.c3_ind = tf.placeholder(tf.int32, [bs//4, fs*fs])

		self.label = tf.placeholder(tf.int32, [bs])
		self.loss_weight = tf.placeholder(tf.float32, [bs])

		label_mask = tf.cast(self.label, tf.bool)

		shape_unpool2 = tf.constant([bs//2, 64])
		shape_unpool1 = tf.constant([bs, 32])

		if 'd' in self.par.input_type:
			if num_input_ch > 1:
				signal_input = tf.concat(input_list, axis=1)
				h_conv1 = lrelu(point_conv('conv1', signal_input, self.c1_ind,
					fs*fs, num_input_ch, 32, extra_chan=self.input_depth1))
			else:
				signal_input = tf.expand_dims(tf.expand_dims(self.input_depth1, axis=2), axis=0)
				h_conv1 = lrelu(conv_2d_layer('conv1', signal_input, 1, 32, 1,
							fs*fs, 1, 1, padding='VALID'))
		else:
			signal_input = tf.concat(input_list, axis=1)
			h_conv1 = lrelu(point_conv('conv1', signal_input, self.c1_ind,
				fs*fs, num_input_ch, 32))

		h_conv1 = tf.squeeze(h_conv1)
		h_conv11 = lrelu(point_conv('conv11', h_conv1, self.c1_ind,
			fs*fs, 32, 32))

		h_pool1 = point_pool(h_conv11, self.p12_ind, self.p12_mask)
		if 'd' in self.par.input_type:
			h_conv2 = lrelu(point_conv('conv2', h_pool1, self.c2_ind, fs*fs, 33, 64,
				extra_chan=self.input_depth2))
		else:
			h_conv2 = lrelu(point_conv('conv2', h_pool1, self.c2_ind, fs*fs, 32, 64))
		h_conv22 = lrelu(point_conv('conv22', h_conv2, self.c2_ind, fs*fs, 64, 64))
		h_pool2 = point_pool(h_conv22, self.p23_ind, self.p23_mask)
		if 'd' in self.par.input_type:
			h_conv3 = lrelu(point_conv('conv3', h_pool2, self.c3_ind, fs*fs, 65, 128,
				extra_chan=self.input_depth3))
		else:
			h_conv3 = lrelu(point_conv('conv3', h_pool2, self.c3_ind, fs*fs, 64, 128))
		h_conv33 = lrelu(point_conv('conv33', h_conv3, self.c3_ind, fs*fs, 128, 64))
		h_unpool2 = point_unpool(h_conv33, self.p23_ind, shape_unpool2)
		uconv2_in = tf.concat([h_conv22, h_unpool2], axis=1)
		h_uconv2 = lrelu(point_conv('uconv2', uconv2_in, self.c2_ind, fs*fs, 128, 64))
		h_uconv22 = lrelu(point_conv('uconv22', h_uconv2, self.c2_ind, fs*fs, 64, 32))
		h_unpool1 = point_unpool(h_uconv22, self.p12_ind, shape_unpool1)
		uconv1_in = tf.concat([h_conv11, h_unpool1], axis=1)
		h_uconv1 = lrelu(point_conv('uconv1', uconv1_in, self.c1_ind, fs*fs, 64, 32))
		h_uconv11 = tf.squeeze(point_conv('uconv11', h_uconv1, self.c1_ind, fs*fs, 32, 32))

		pred_input = tf.expand_dims(tf.expand_dims(h_uconv11, axis=1), axis=0)
		h_pred = tf.squeeze(conv_2d_layer('pred1', pred_input, 32, self.par.d_par.num_classes, 1, 1, 1, 1))

		self.output = tf.argmax(h_pred, axis=1, output_type=tf.int32)

		masked_output = tf.boolean_mask(h_pred, label_mask)
		masked_label = tf.boolean_mask(self.label, label_mask)
		masked_weights = tf.boolean_mask(self.loss_weight, label_mask)

		tr_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "")

		self.loss = tf.reduce_mean(tf.multiply(masked_weights,
			tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_label,
														   logits=masked_output)))
		self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

		correct_prediction = tf.equal(tf.argmax(masked_output, axis=1, output_type=tf.int32), masked_label)
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.test_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
		self.test_loss_summary = tf.summary.scalar("accuracy", self.test_loss_placeholder)

		self.train_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
		self.train_loss_summary = tf.summary.scalar("train_loss", self.train_loss_placeholder)

		curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
		self.writer = tf.summary.FileWriter(os.path.join(self.par.log_dir, curr_time))

		self.saver = tf.train.Saver(tr_var, max_to_keep=self.par.max_snapshots)

	def initialize_model(self):
		self.sess.run(tf.global_variables_initializer())

	def save_snapshot(self):
		self.saver.save(self.sess, os.path.join(self.par.snapshot_dir, 'model'),
			global_step=self.training_step)

	def load_snapshot(self):
		snapshot_name = tf.train.latest_checkpoint(self.par.snapshot_dir)
		if snapshot_name is not None:
			model_file_name = os.path.basename(snapshot_name)
			print("Loading snapshot " + model_file_name)
			itn = int(model_file_name.split('-')[1])
			self.training_step = itn
			self.saver.restore(self.sess, snapshot_name)

	def train(self):
		bs = self.par.batch_size
		for iter_i in range(self.training_step, self.par.max_iter_count):
			if (iter_i > 0) and (iter_i % self.par.reload_iter == 0):
				self.load_data("train")

			if iter_i % self.par.test_iter == 0:
				self.validate(iter_i)

			b = self.get_training_batch(iter_i)

			if b.num_points() > bs:
				continue

			out = self.sess.run([self.train_step, self.loss, self.output], feed_dict=self.get_feed_dict(b))
			print(str(iter_i) + " : " + str(out[1]))

			summary = self.sess.run(self.train_loss_summary,
				feed_dict={self.train_loss_placeholder: out[1]})
			self.writer.add_summary(summary, iter_i)

	def validate(self, step):
		pixel_count = 0
		acc = []
		pix = []
		bs = self.par.batch_size
		for b in self.validation_batches:
			out = self.sess.run([self.accuracy, self.output], feed_dict=self.get_feed_dict(b))
			valid_out = np.multiply(out[1], np.asarray(expand_dim_to_batch1(b.labels[0], bs), dtype=bool))
			acc.append(out[0])
			pix.append(np.count_nonzero(b.labels[0]))
			pixel_count += np.count_nonzero(b.labels[0])

		avg_acc = 0.0
		for i in range(0, len(acc)):
			avg_acc += acc[i] * pix[i] / pixel_count
		print("Accuracy: " + str(avg_acc))

		if avg_acc > self.best_accuracy:
			self.best_accuracy = avg_acc
			self.save_snapshot()

		summary = self.sess.run(self.test_loss_summary,
			feed_dict={self.test_loss_placeholder: avg_acc})
		self.writer.add_summary(summary, step)

	def test(self):
		scan_id = 0
		cs = self.par.cube_size
		print("Testing...")
		for val_scan in self.test_data:
			if self.par.data_sampling_type == 'full':
				scan_batches = [get_batch_from_full_scan(val_scan, self.par.num_scales, self.par.d_par.class_weights)]
			else:
				global scan
				min_bound = val_scan.clouds[0].get_min_bound() - cs[0] * 0.5
				max_bound = val_scan.clouds[0].get_max_bound() + cs[0] * 0.5
				scan = val_scan
				rad = self.par.valid_rad
				points = []
				x_s = min_bound[0] + rad / 2.0
				while x_s < max_bound[0] - rad:
					y_s = min_bound[1] + rad / 2.0
					while y_s < max_bound[1] - rad:
						z_s = min_bound[2] + rad / 2.0
						while z_s < max_bound[2] - rad:
							if val_scan.has_points([x_s, y_s, z_s], rad):
								points.append([x_s, y_s, z_s])
							z_s += rad
						y_s += rad
					x_s += rad
				arr_size = len(points)
				print("Number of test batches: " + str(arr_size))
				print("Loading batches...")
				scan_batches = get_batch_array(val_scan, self.par, points)
				print("Done.")

			for b in scan_batches:
				out = self.sess.run(self.output, feed_dict=self.get_feed_dict(b))
				valid_out = np.multiply(out, np.asarray(expand_dim_to_batch1(b.labels[0], self.par.batch_size), dtype=bool))
				if self.par.data_sampling_type == 'full':
					val_scan.assign_labels(valid_out)
				else:
					val_scan.assign_labels_part(valid_out, b.index_maps[0])

			make_dir(os.path.join(self.par.output_dir, self.test_scans[scan_id]))
			val_scan.save(os.path.join(self.par.output_dir, self.test_scans[scan_id]))
			print(self.test_scans[scan_id])
			scan_id += 1


def run_net(config, mode):
	par = param(config)
	tf.reset_default_graph()
	nn = model(par)

	make_dir(par.log_dir)
	make_dir(par.output_dir)
	make_dir(par.snapshot_dir)

	if mode == "train":
		nn.load_data("test")
		nn.load_data("train")
		nn.build_model(par.batch_size)
		nn.precompute_validation_batches()
		nn.initialize_model()
		nn.load_snapshot()
		nn.train()
	elif mode == "test":
		nn.load_data("test")
		nn.build_model(par.batch_size)
		nn.initialize_model()
		nn.load_snapshot()
		nn.test()

	return 0