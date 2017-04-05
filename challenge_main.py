#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import sys
from os.path import join

from sklearn.utils import shuffle
from utils import conv_layer, fc_layer
from utils import Cursors


#############################################
############### IMPORT DATA #################
#############################################

data_path = '..'

images_train_fname    = join(data_path, 'data_train.bin')
templates_train_fname = join(data_path, 'fv_train.bin')

images_valid_fname    = join(data_path, 'data_valid.bin')
templates_valid_fname = join(data_path, 'fv_valid.bin')

images_test_fname     = join(data_path, 'data_test.bin')

# number of images
num_train_images = 100000
num_valid_images = 10000
num_test_images  = 10000

# size of the images 48*48 pixels in gray levels
image_dim = 48
image_size = image_dim ** 2
img_range = 255

# dimension of the templates
template_dim = 128

# read the training files
with open(templates_train_fname, 'rb') as f:
    train_template_data = np.fromfile(f, dtype=np.float32, count=num_train_images * template_dim)
    train_template_data = train_template_data.reshape(num_train_images, template_dim)

with open(images_train_fname, 'rb') as f:
    train_image_data = np.fromfile(f, dtype=np.uint8, count=num_train_images * image_size).astype(np.float32)
    train_image_data = train_image_data.reshape(num_train_images, image_size)

# read the validation files
with open(templates_valid_fname, 'rb') as f:
    valid_template_data = np.fromfile(f, dtype=np.float32, count=num_valid_images * template_dim)
    valid_template_data = valid_template_data.reshape(num_valid_images, template_dim)

with open(images_valid_fname, 'rb') as f:
    valid_image_data = np.fromfile(f, dtype=np.uint8, count=num_valid_images * image_size).astype(np.float32)
    valid_image_data = valid_image_data.reshape(num_valid_images, image_size)

# read the test file
with open(images_test_fname, 'rb') as f:
    test_image_data = np.fromfile(f, dtype=np.uint8, count=num_test_images * image_size).astype(np.float32)
    test_image_data = test_image_data.reshape(num_test_images, image_size)


######### data pre-processing
train_image_data_mean = np.mean(train_image_data) 

train_imgs = (train_image_data - train_image_data_mean) / img_range
valid_imgs = (valid_image_data - train_image_data_mean) / img_range
test_imgs = (test_image_data - train_image_data_mean) / img_range


#####################################################################################################################
#####################################################################################################################
# Params

nb_img_train, nb_features = train_imgs.shape
nb_img_valid, _ = valid_imgs.shape
nb_img_test, _ = test_imgs.shape

_, predictions_size = train_template_data.shape

max_epoch = 300
batch_train = 500
batch_test = 500

epoch_step = batch_train / nb_img_train
nbiter_epoch = np.floor(nb_img_train / batch_train)
nb_max_iter = np.floor(max_epoch / epoch_step)

dropout = 0.5

summary_dir = '../tensorlog'
folder_name = 'epoch_%.1f_dp_%i_batch_norm' % (dropout, max_epoch)
full_dir = join(summary_dir, folder_name)

validation_log_frequency = 5
validation_log_frequency_iter = np.floor(validation_log_frequency / epoch_step).astype(int)

evaluation_log_frequency = 20
evaluation_log_frequency_iter = np.floor(evaluation_log_frequency / epoch_step).astype(int)

training_log_frequency = 0.5
training_log_frequency_iter = np.floor(training_log_frequency / epoch_step).astype(int)

reshuffling_frequency = 3.0
reshuffling_frequency_iter = np.floor(reshuffling_frequency / epoch_step).astype(int)

nb_montecarlo_predictions = 50

inital_lr = 5e-6

np.random.seed(666)
tf.set_random_seed(10)
#####################################################################################################################
#####################################################################################################################

#### Placeholders
with tf.name_scope('input'):
	x_ = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, template_dim], name='y-input')
	keep_prob = tf.placeholder(tf.float32, name='dropout')
	is_training = tf.placeholder(np.float32, name='is-training')

placeholder_dict = {'x_': x_, 'y_': y_, 'keep_prob': keep_prob, 'is-training': is_training}

#############################################
############### THE NETWORK #################
#############################################

stride = 1
filter_size = 3
filter_nb_1 = 10
filter_nb_2 = 15
filter_nb_3 = 20

hidden1 = conv_layer(x_, [filter_size, filter_size, 1, filter_nb_1], 'conv-1', stride, keep_prob, is_training)
hidden2 = conv_layer(hidden1, [filter_size, filter_size, filter_nb_1, filter_nb_1], 'conv-2', stride, keep_prob, is_training)

pool3 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC', name=None)

hidden4 = conv_layer(pool3, [filter_size, filter_size, filter_nb_1, filter_nb_2], 'conv-4', stride, keep_prob, is_training)
hidden5 = conv_layer(hidden4, [filter_size, filter_size, filter_nb_2, filter_nb_2], 'conv-5', stride, keep_prob, is_training)

pool6 = tf.nn.max_pool(hidden5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC', name=None)

hidden7 = conv_layer(pool6, [filter_size, filter_size, filter_nb_2, filter_nb_3], 'conv-7', stride, keep_prob, is_training)
hidden8 = conv_layer(hidden7, [filter_size, filter_size, filter_nb_3, filter_nb_3], 'conv-8', stride, keep_prob, is_training)

pool9 = tf.nn.max_pool(hidden8, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC', name=None)

hidden10 = conv_layer(pool9, [filter_size, filter_size, filter_nb_3, filter_nb_3], 'conv-10', stride, keep_prob, is_training)
hidden11 = conv_layer(hidden10, [filter_size, filter_size, filter_nb_3, filter_nb_3], 'conv-11', stride, keep_prob, is_training)

pool12 = tf.nn.max_pool(hidden11, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC', name=None)

hidden13 = tf.reshape(pool12, shape=[-1, 3 * 3 * filter_nb_3])

y = fc_layer(hidden13, [3 * 3 * filter_nb_3, template_dim], 'fc-final', keep_prob, act=None)

#############################################
################ THE LOSS ###################
#############################################

""" Loss for regression """
with tf.name_scope('training'):
	euclidean_loss = tf.reduce_mean(tf.square(y - y_))

tf.summary.scalar('train_euclidean_loss', euclidean_loss)


""" Learning rate """
with tf.name_scope('learning_rate'):
    global_step = tf.Variable(0, trainable=False)
    decay_epoch = 20
    learning_rate = tf.train.exponential_decay(inital_lr, global_step, np.floor(decay_epoch * nbiter_epoch), 0.97, staircase=True)

tf.summary.scalar('learning_rate_summary', learning_rate)


""" Optimizer """
with tf.name_scope('opt-training'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(euclidean_loss, global_step=global_step)


merged_train_summary = tf.summary.merge_all()


with tf.name_scope('validation'):
	validation_loss = tf.placeholder(tf.float32, name='loss')

summary_validation_loss = tf.summary.scalar('validation_euclidean_loss', validation_loss)


############ IMAGE SUMMARIES
with tf.name_scope('high_variance_images'):
	training_high_variance_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
	summary_high_variance_images = tf.summary.image('high_variance', training_high_variance_images, 5)

with tf.name_scope('low_variance_images'):
	training_low_variance_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
	summary_low_variance_images = tf.summary.image('low_variance', training_low_variance_images, 5)

with tf.name_scope('high_error_images'):
	training_high_error_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
	summary_high_error_images = tf.summary.image('high_error', training_high_error_images, 5)

with tf.name_scope('low_error_images'):
	training_low_error_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
	summary_low_error_images = tf.summary.image('low_error', training_low_error_images, 5)


summary_images = tf.summary.merge([summary_high_variance_images, 
								summary_low_variance_images,
								summary_high_error_images,
								summary_low_error_images])


##########################################################################################################
if tf.gfile.Exists(full_dir):
	var = input('The folder {:s} already exists.' + ' Would you like to overwrite it ?\n' + 'yes(y), no(n): '.format(full_dir))

	if not var in ['y', 'yes']:
	    sys.exit()
	
	tf.gfile.DeleteRecursively(full_dir)
	tf.gfile.MakeDirs(full_dir)
else:
	tf.gfile.MakeDirs(full_dir)

sess = tf.Session()

train_writer = tf.summary.FileWriter(full_dir + '/train', sess.graph)
validation_writer = tf.summary.FileWriter(full_dir + '/validation')

init = tf.global_variables_initializer()
sess.run(init)

##########################################################################################################
##########################################################################################################
rng = np.random.RandomState(42)

train_imgs = train_imgs.reshape([-1, image_dim, image_dim, 1]) 
valid_imgs = valid_imgs.reshape([-1, image_dim, image_dim, 1])
test_imgs = test_imgs.reshape([-1, image_dim, image_dim, 1])

X_train, y_train = shuffle(train_imgs, train_template_data, random_state=42)

cursors = Cursors()


def feed_func(batch_size, mode='train', placeholder_dict=placeholder_dict, cursors=cursors):

	if mode == 'train':

		tmp_cur = cursors.train_current_pos
		ind_batch = np.mod(tmp_cur + np.arange(batch_size), nb_img_train).astype(int)
		
		X_tmp = X_train[ind_batch]
		y_tmp = y_train[ind_batch]

		cursors.train_current_pos = ind_batch[-1] + 1

		is_training_tmp = 1.0

	elif mode == 'valid':

		tmp_cur = cursors.validation_current_pos
		ind_batch = np.mod(tmp_cur + np.arange(batch_size), nb_img_valid).astype(int)
		
		X_tmp = valid_imgs[ind_batch]
		y_tmp = valid_template_data[ind_batch]

		cursors.validation_current_pos = ind_batch[-1] + 1

		is_training_tmp = 0.0

	# non shuffled dataset
	elif mode == 'eval':

		tmp_cur = cursors.eval_current_pos
		ind_batch = np.mod(tmp_cur + np.arange(batch_size), nb_img_train).astype(int)
		
		X_tmp = train_imgs[ind_batch]
		y_tmp = train_template_data[ind_batch]

		cursors.eval_current_pos = ind_batch[-1] + 1

		is_training_tmp = 0.0

	return {placeholder_dict['x_']: X_tmp, placeholder_dict['y_']: y_tmp, placeholder_dict['keep_prob']: dropout, placeholder_dict['is-training']: is_training_tmp}



##################### TRAINING LOOP #####################"""
i = 0
nb_iter_validation = np.ceil(nb_img_valid / batch_test)
nb_iter_evaluation = np.ceil(nb_img_train / batch_test)

while i < nb_max_iter:

	####################### VALIDATION MODE ############################
	if ((np.mod(i, validation_log_frequency_iter) == 0) & (not i == 0)):

		cursors.validation_current_pos = 0

		montecarlo_samples_validation = np.zeros((nb_img_valid, template_dim, nb_montecarlo_predictions), dtype=np.float32)
		
		for jj in np.arange(nb_iter_validation):
			ind_tmp = np.mod(jj * batch_test + np.arange(batch_test), nb_img_valid).astype(int)
			feed_dict = feed_func(batch_test, mode='valid')

			for kk in np.arange(nb_montecarlo_predictions):
				mc_sample = sess.run(y, feed_dict=feed_dict)
				montecarlo_samples_validation[ind_tmp, :, kk] = mc_sample

		montecarlo_predictions_validation = np.mean(montecarlo_samples_validation, axis=2)
		
		validation_score = np.mean((montecarlo_predictions_validation - valid_template_data)** 2)

		valid_sum = sess.run(summary_validation_loss, feed_dict={validation_loss:validation_score})
		validation_writer.add_summary(valid_sum, i)

		print('{:.1f} epoch || validation score: {:.4e}'.format(i * epoch_step, validation_score))


	####################### TRAIN MODE ############################
	if ((np.mod(i, training_log_frequency_iter) == 0) & (not i == 0)):

		train_sum, _, loss = sess.run([merged_train_summary, train_op, euclidean_loss], feed_dict=feed_func(batch_train, mode='train'))
		
		train_writer.add_summary(train_sum, i)

		####
		print('{:.1f} epoch || training loss: {:.4e}'.format(i * epoch_step, loss))

	else:
		_ = sess.run(train_op, feed_dict=feed_func(batch_train, mode='train'))


	##################### EVAL ON TRAIN DATASET ###################
	if ((np.mod(i, evaluation_log_frequency_iter) == 0) & (not i == 0)):


		cursors.eval_current_pos = 0
		
		montecarlo_samples_evaluation = np.zeros((nb_img_train, template_dim, nb_montecarlo_predictions), dtype=np.float32)
		
		for jj in np.arange(nb_iter_evaluation):
			ind_tmp = np.mod(jj * batch_test + np.arange(batch_test), nb_img_train).astype(int)
			feed_dict = feed_func(batch_test, mode='eval')

			for kk in np.arange(nb_montecarlo_predictions):
				mc_sample = sess.run(y, feed_dict=feed_dict)
				montecarlo_samples_evaluation[ind_tmp, :, kk] = mc_sample

		montecarlo_predictions_evaluation = np.mean(montecarlo_samples_evaluation, axis=2)
		#centred_prediction_evaluation = montecarlo_samples_evaluation - montecarlo_predictions_evaluation.reshape(-1, -1, 1)

		train_squared_error = np.sum((montecarlo_predictions_evaluation - train_template_data)** 2, axis=1)
		full_train_loss = np.mean((montecarlo_predictions_evaluation - train_template_data)** 2)


		sorted_ind = np.argsort(train_squared_error)
		high_error_ind = sorted_ind[-5:]
		low_error_ind = sorted_ind[:5]

		feed_images = {training_high_error_images: train_imgs[high_error_ind], 
					training_low_error_images:train_imgs[low_error_ind]}

		sum_high_err_img, sum_low_err_img=sess.run([summary_high_error_images, summary_low_error_images], feed_dict=feed_images)

		train_writer.add_summary(sum_high_err_img, i)
		train_writer.add_summary(sum_low_err_img, i)

		print('{:.1f} epoch || full training loss: {:.4e}'.format(i * epoch_step, full_train_loss))

	if np.mod(i, reshuffling_frequency_iter) == 0:
		print('Shuffling training data')
		train_imgs, train_template_data = shuffle(train_imgs, train_template_data, random_state=42)


	i += 1




















































