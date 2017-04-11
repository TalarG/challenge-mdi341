#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import sys
from os.path import join

from sklearn.utils import shuffle
from utils import conv_layer, fc_layer
from utils import Cursors
from sklearn.decomposition import PCA

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
train_image_data_mean = np.mean(train_image_data, axis=1).reshape(-1, 1)
train_image_data_std = np.std(train_image_data, axis=1).reshape(-1, 1)

valid_image_data_mean = np.mean(valid_image_data, axis=1).reshape(-1, 1)
valid_image_data_std = np.std(valid_image_data, axis=1).reshape(-1, 1)

test_image_data_mean = np.mean(test_image_data, axis=1).reshape(-1, 1) 
test_image_data_std = np.std(test_image_data, axis=1).reshape(-1, 1)

train_imgs = (train_image_data - train_image_data_mean) / train_image_data_std
valid_imgs = (valid_image_data - valid_image_data_mean) / valid_image_data_std
test_imgs = (test_image_data - test_image_data_std) / test_image_data_std

#####################################################################################################################
#####################################################################################################################
# Params
nb_img_train, nb_features = train_imgs.shape
nb_img_valid, _ = valid_imgs.shape
nb_img_test, _ = test_imgs.shape

_, predictions_size = train_template_data.shape

max_epoch = 3000
batch_train = 800
batch_test = 2000

epoch_step = batch_train / nb_img_train
nbiter_epoch = np.floor(nb_img_train / batch_train)
nb_max_iter = np.floor(max_epoch / epoch_step)

dropout = 0.90
decay_epoch = 100
decay_factor = 0.9
inital_lr = 3e-3
batch_norm = False
nb_montecarlo_predictions = 80

pre_processing = True
power_pca = - 1 / 5
nb_kept_components = 2000

summary_dir = '../tensorlog'
folder_name = 'epoch_%i_dp_%.2f_nbmcdp_%i' % (max_epoch, dropout, nb_montecarlo_predictions)

if pre_processing:
	folder_name += '_preprocess_%.1f_%i' % (-power_pca, nb_kept_components)
if batch_norm:
	folder_name += '_batchnorm'

folder_name += '_0'

full_dir = join(summary_dir, folder_name)

validation_log_frequency = 20
evaluation_log_frequency = 100
training_log_frequency = 0.5
reshuffling_frequency = 3.0

validation_log_frequency_iter = np.floor(validation_log_frequency / epoch_step).astype(int)
evaluation_log_frequency_iter = np.floor(evaluation_log_frequency / epoch_step).astype(int)
training_log_frequency_iter = np.floor(training_log_frequency / epoch_step).astype(int)
reshuffling_frequency_iter = np.floor(reshuffling_frequency / epoch_step).astype(int)

np.random.seed(666)
tf.set_random_seed(10)

nb_display_images = 8

#####################################################################################################################
#####################################################################################################################
if pre_processing:
	pca = PCA(svd_solver='randomized', n_components=nb_kept_components)
	pca.fit(train_imgs)

	pca_preprocess = lambda x: x.dot(pca.components_.T).dot(pca.components_ * np.power(pca.explained_variance_, power_pca).reshape(-1,1))

	train_imgs = pca_preprocess(train_imgs)
	valid_imgs = pca_preprocess(valid_imgs)
	test_imgs = pca_preprocess(test_imgs)

'''
indices_components_loss = np.array([28,1,105,59,46,15,55,107,83,75,109,16,82,106,25,18,93,89,97,34,92,64,61,48,125,112,49,113,87,33,56,62,96,78,86,42,51,50,41,76,67,20,60,70,110,26,32,99,104,17,43,77,57,101,35,11,91,7,58,8,54,88,19,73,98,38,12,53,2,94,102,127,66,122,126,37,90,24,95,6,14,103,31,68,74,65,10,111,114,27,124,36,39,79,115,72,3,119,22,45,23,100,108,52,117,30,21,44,84,13,69,120,9,40,81,118,85,116,71,80,47,121,63,4,5,0,123,29])
weights_loss = np.ones(128)
weights_loss[indices_components_loss[-20:]] = 1
weights_loss = weights_loss.reshape(1, -1)
'''
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
filter_nb_3 = 25

activation_func = tf.nn.relu

hidden1 = conv_layer(x_, [filter_size, filter_size, 1, filter_nb_1], 'conv-1', stride, keep_prob, is_training, act=activation_func)
hidden2 = conv_layer(hidden1, [filter_size, filter_size, filter_nb_1, filter_nb_1], 'conv-2', stride, keep_prob, is_training, act=activation_func)

pool3 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC', name=None)

hidden4 = conv_layer(pool3, [filter_size, filter_size, filter_nb_1, filter_nb_2], 'conv-4', stride, keep_prob, is_training, act=activation_func)
hidden5 = conv_layer(hidden4, [filter_size, filter_size, filter_nb_2, filter_nb_2], 'conv-5', stride, keep_prob, is_training, act=activation_func)

pool6 = tf.nn.max_pool(hidden5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC', name=None)

hidden7 = conv_layer(pool6, [filter_size, filter_size, filter_nb_2, filter_nb_3], 'conv-7', stride, keep_prob, is_training, act=activation_func)
hidden8 = conv_layer(hidden7, [filter_size, filter_size, filter_nb_3, filter_nb_3], 'conv-8', stride, keep_prob, is_training, act=activation_func)

pool9 = tf.nn.max_pool(hidden8, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC', name=None)

hidden10 = conv_layer(pool9, [filter_size, filter_size, filter_nb_3, filter_nb_3], 'conv-10', stride, keep_prob, is_training, act=activation_func)
hidden11 = conv_layer(hidden10, [filter_size, filter_size, filter_nb_3, filter_nb_3], 'conv-11', stride, keep_prob, is_training, act=activation_func)

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
    learning_rate = tf.train.exponential_decay(inital_lr, global_step, np.floor(decay_epoch * nbiter_epoch), decay_factor, staircase=True)

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


#### Training
with tf.name_scope('training-high-variance-images'):
	training_high_variance_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
summary_training_high_variance_images = tf.summary.image('training-high-variance', training_high_variance_images, nb_display_images)

with tf.name_scope('training-low-variance-images'):
	training_low_variance_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
summary_training_low_variance_images = tf.summary.image('training-low-variance', training_low_variance_images, nb_display_images)

with tf.name_scope('training-high-error-images'):
	training_high_error_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
summary_training_high_error_images = tf.summary.image('training-high-error', training_high_error_images, nb_display_images)

with tf.name_scope('training-low-error-images'):
	training_low_error_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
summary_training_low_error_images = tf.summary.image('training-low-error', training_low_error_images, nb_display_images)


summary_training_images = tf.summary.merge([summary_training_high_variance_images, 
								summary_training_low_variance_images,
								summary_training_high_error_images,
								summary_training_low_error_images])

#### Validation
with tf.name_scope('validation-high-variance-images'):
	validation_high_variance_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
summary_validation_high_variance_images = tf.summary.image('validation-high-variance', validation_high_variance_images, nb_display_images)

with tf.name_scope('validation-low-variance-images'):
	validation_low_variance_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
summary_validation_low_variance_images = tf.summary.image('validation-low-variance', validation_low_variance_images, nb_display_images)

with tf.name_scope('validation-high-error-images'):
	validation_high_error_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
summary_validation_high_error_images = tf.summary.image('validation-high-error', validation_high_error_images, nb_display_images)

with tf.name_scope('validation-low-error-images'):
	validation_low_error_images = tf.placeholder(tf.float32, [None, image_dim, image_dim, 1])
summary_validation_low_error_images = tf.summary.image('validation-low-error', validation_low_error_images, nb_display_images)


summary_validation_images = tf.summary.merge([summary_validation_high_variance_images, 
								summary_validation_low_variance_images,
								summary_validation_high_error_images,
								summary_validation_low_error_images])


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

	return {placeholder_dict['x_']: X_tmp, placeholder_dict['y_']: y_tmp, 
			placeholder_dict['keep_prob']: dropout, placeholder_dict['is-training']: is_training_tmp}



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
		
		validation_squared_error = np.sum((montecarlo_predictions_validation - valid_template_data)** 2, axis=1)
		validation_score = np.mean((montecarlo_predictions_validation - valid_template_data)** 2)


		sorted_ind = np.argsort(validation_squared_error)
		high_error_ind = sorted_ind[-nb_display_images:]
		low_error_ind = sorted_ind[:nb_display_images]

		feed_images = {validation_high_error_images: valid_imgs[high_error_ind], 
					validation_low_error_images:valid_imgs[low_error_ind]}

		sum_high_err_img, sum_low_err_img = sess.run([summary_validation_high_error_images, 
												summary_validation_low_error_images], feed_dict=feed_images)

		validation_writer.add_summary(sum_high_err_img, i)
		validation_writer.add_summary(sum_low_err_img, i)


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
		high_error_ind = sorted_ind[-nb_display_images:]
		low_error_ind = sorted_ind[:nb_display_images]

		feed_images = {training_high_error_images: train_imgs[high_error_ind], 
					training_low_error_images:train_imgs[low_error_ind]}

		sum_high_err_img, sum_low_err_img=sess.run([summary_training_high_error_images, 
												summary_training_low_error_images], feed_dict=feed_images)

		train_writer.add_summary(sum_high_err_img, i)
		train_writer.add_summary(sum_low_err_img, i)

		print('{:.1f} epoch || full training loss: {:.4e}'.format(i * epoch_step, full_train_loss))

	
	if np.mod(i, reshuffling_frequency_iter) == 0:
		print('Shuffling training data')
		train_imgs, train_template_data = shuffle(train_imgs, train_template_data, random_state=42)

	i += 1


##################### PREDICT ON TETS DATASET ###################
montecarlo_samples_test = np.zeros((nb_img_test, template_dim, nb_montecarlo_predictions), dtype=np.float32)
nb_iter_test = np.ceil(nb_img_test / batch_test).astype(int)

for jj in np.arange(nb_iter_test):
	ind_tmp = np.mod(jj * batch_test + np.arange(batch_test), nb_img_test).astype(int)
	feed_dict = {placeholder_dict['x_']: test_imgs[ind_tmp], placeholder_dict['keep_prob']: dropout, 
					placeholder_dict['is-training']: 1.0}
	for kk in np.arange(nb_montecarlo_predictions):
		mc_sample = sess.run(y, feed_dict=feed_dict)
		montecarlo_samples_test[ind_tmp, :, kk] = mc_sample

montecarlo_predictions_test = np.mean(montecarlo_samples_test, axis=2)


output_file_name = join('..', folder_name + '_template_pred.bin' )
f = open(output_file_name, 'wb')
for i in range(nb_img_test):
    f.write(montecarlo_predictions_test[i, :])

f.close()















































