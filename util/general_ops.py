import tensorflow as tf

def lrelu(x, alpha=0.1):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def fc_layer(name, in_tensor, in_size, out_size, stddev=0.05):
	with tf.variable_scope(name):
		W = tf.get_variable('W', [in_size, out_size], initializer=tf.truncated_normal_initializer(stddev=stddev))
		b = tf.get_variable('b', [out_size], initializer=tf.constant_initializer(0.0))
		return tf.matmul(in_tensor, W) + b

def conv_2d_layer(name, in_tensor, in_ch, out_ch, k_h, k_w, s_h, s_w, stddev=0.01, initial_w=None, padding='SAME'):
	with tf.variable_scope(name):
		W = tf.get_variable('W', [k_h, k_w, in_ch, out_ch], initializer=tf.contrib.layers.xavier_initializer(True))
		conv = tf.nn.conv2d(in_tensor, W, strides=[1, s_h, s_w, 1], padding=padding)
		b = tf.get_variable('b', [out_ch], initializer=tf.constant_initializer(0.01))
		return tf.reshape(tf.nn.bias_add(conv, b), tf.shape(conv))
