import tensorflow as tf

#### Functions
def conv_layer(input_tensor, shape, layer_name, stride, keep_prob):
	"""
	Creation of convolution layer followed by a Relu by default
	"""

	# Adding a name scope ensures logical grouping of the layers in the graph.
	with tf.variable_scope(layer_name):

		# This Variable will hold the state of the weights for the layer
		#with tf.variable_scope('weights'):
		weights = tf.get_variable(name='weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer())

		#with tf.variable_scope('biases'):
		biases = tf.get_variable(name='bias', shape=shape[3], initializer=tf.constant_initializer(0.02))

		with tf.name_scope('pre-activation'):
		    preactivate = tf.nn.conv2d(input_tensor, weights, strides=[1, stride, stride, 1], padding='SAME') + biases

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







































