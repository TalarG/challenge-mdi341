import tensorflow as tf

#### Functions
def conv_layer(input_tensor, shape, layer_name, stride, keep_prob, is_training):
	"""
	Creation of convolution layer followed by a Relu by default
	"""

	# Adding a name scope ensures logical grouping of the layers in the graph.
	with tf.variable_scope(layer_name):

		decay = 0.999
		epsilon = 1e-3

		weights = tf.get_variable(name='weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer())

		biases = tf.get_variable(name='bias', shape=shape[3], initializer=tf.constant_initializer(0.02))

		with tf.name_scope('pre-activation'):
		    preactivate = tf.nn.conv2d(input_tensor, weights, strides=[1, stride, stride, 1], padding='SAME') + biases

		#import ipdb; ipdb.set_trace()
		with tf.name_scope('batch-normalization'):
			scale = tf.get_variable(name='scale', shape=preactivate.shape[1:], initializer=tf.constant_initializer(1.0))
			beta = tf.get_variable(name='beta', shape=preactivate.shape[1:], initializer=tf.constant_initializer(0.0))
			pop_mean = tf.get_variable(name='pop_mean', shape=preactivate.shape[1:], initializer=tf.constant_initializer(1.0), trainable=False)
			pop_var = tf.get_variable(name='pop-var', shape=preactivate.shape[1:], initializer=tf.constant_initializer(0.0), trainable=False)

			if is_training == 1.0:
				batch_mean, batch_var = tf.nn.moments(preactivate, axes = [0])
				train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
				train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

				with tf.control_dependencies([train_mean, train_var]):
					batch_norm_preactivate = tf.nn.batch_normalization(preactivate, batch_mean, batch_var, beta, scale, epsilon)

			else:
				batch_norm_preactivate = tf.nn.batch_normalization(preactivate, pop_mean, pop_var, beta, scale, epsilon)

		with tf.name_scope('activation'):
			activations = tf.nn.relu(preactivate)
		
		act_dp = tf.nn.dropout(activations, keep_prob)
		
		return act_dp



def fc_layer(input_tensor, shape, layer_name, keep_prob, act=tf.nn.relu):
	""" Fully connected layer
	It does a matrix multiply, bias add, and then uses relu to nonlinearize.
	It also sets up name scoping so that the resultant graph is easy to read, and
	adds a number of summary ops.
	"""

	# Adding a name scope ensures logical grouping of the layers in the graph.
	with tf.variable_scope(layer_name):

		# This Variable will hold the state of the weights for the layer
		#with tf.variable_scope('weights'):
		weights = tf.get_variable(name='weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer())

		#with tf.variable_scope('biases'):
		biases = tf.get_variable(name='bias', shape=shape[1], initializer=tf.constant_initializer(0.02))

		with tf.name_scope('pre-activation'):
		    preactivate = tf.matmul(input_tensor, weights) + biases

		if (act==None):
		    return preactivate

		else:
			with tf.name_scope('activation'):
				activations = act(preactivate)

			act_dp = tf.nn.dropout(activations, keep_prob)
			return act_dp


class Cursors(object):
	
	train_current_pos = 0
	validation_current_pos = 0
	eval_current_pos = 0

	def __init():
		return None

































