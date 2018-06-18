import tensorflow as tf

def point_conv(name, input_tensor, index_tensor,
			   filter_size, in_channels, out_channels, stddev=0.05, extra_chan=None):
	with tf.variable_scope(name):
		W = tf.get_variable('W', [1, filter_size, in_channels, out_channels],
				initializer=tf.contrib.layers.xavier_initializer(True))
		b = tf.get_variable('b', [out_channels], initializer=tf.constant_initializer(0.01))
		conv_input = tf.gather_nd(input_tensor, tf.expand_dims(index_tensor, axis=2))
		if extra_chan is not None:
			conv_input = tf.concat([conv_input, tf.expand_dims(extra_chan, axis=2)], axis=2)
		conv_input = tf.expand_dims(conv_input, axis=0)
		conv_output = tf.nn.conv2d(conv_input, W, strides=[1, 1, 1, 1], padding='VALID')
		conv_output = tf.nn.bias_add(conv_output, b)
		return tf.squeeze(conv_output, [0, 2])

def point_pool(input_tensor, index_tensor, pool_mask, pool_type='MAX'):
	pool_input = tf.gather_nd(input_tensor, tf.expand_dims(index_tensor, axis=2))
	broadcast_mask = tf.tile(tf.expand_dims(pool_mask, axis=2),
							 tf.stack([1, 1, tf.shape(pool_input)[2]]))
	return tf.reduce_sum(pool_input * broadcast_mask, axis=1)

def point_unpool(input_tensor, index_tensor, output_shape, elem_count=8):
	out = []
	for i in range(0, elem_count):
		out.append(tf.scatter_nd(tf.slice(index_tensor, [0, i], [-1, 1]), input_tensor, output_shape))
	return tf.add_n(out)