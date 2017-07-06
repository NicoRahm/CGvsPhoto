from CGvsPhoto import Model
from texture import Texture_model, load_model, compute_dense_sift, compute_fisher

from multiprocessing import Pool

from functools import partial

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import tensorflow as tf


def compute_proba_test(noise_model, texture_model, data, 
				  	   nb_batch, batch_size = 50): 
	
	data_test = []
	for i in range(nb_batch):
		print('Getting batch ' + str(i+1) + '/' + str(nb_batch))
		images_batch, y_batch = data.get_batch_test(batch_size = batch_size,
													crop = False)
		data_test.append([images_batch, y_batch])


	# Texture features 

	features_test_texture = np.empty([nb_batch*batch_size, 128, texture_model.nb_mini_patch])
	y_test = np.empty([nb_batch*batch_size, ])

	pool = Pool()  

	to_compute = [i for i in range(nb_batch)]
	result = pool.starmap(partial(compute_dense_sift, 
								  batch_size = batch_size, 
								  nb_mini_patch = texture_model.nb_mini_patch, 
								  nb_batch = nb_batch,
								  only_green = texture_model.only_green),
								  zip(data_test, to_compute)) 


	
	index = 0
	for i in range(len(result)):
		features_test_texture[index:index+batch_size] = result[i][0]
		y_test[index:index+batch_size] = result[i][1]

		index+=batch_size

	del(result)

	print('Dimension reduction...')
	features_test_PCA = np.empty([nb_batch*batch_size, texture_model.keep_PCA, texture_model.nb_mini_patch])
	for i in range(texture_model.nb_mini_patch):
		# normalize(features_test[:,:,i])
		# features_test_PCA[:,:,i] = pca.transform(features_test[:,:,i])
		features_test_PCA[:,:,i] = texture_model.PCAs[i].transform(features_test_texture[:,:,i])

	del(features_test_texture)

	print('Computing Fisher vectors...')
	fisher_test = compute_fisher(features_test_PCA, texture_model.gmm)

	del(features_test_PCA)

	print('Prediction for textures...')
	y_pred_texture = texture_model.clf_svm.predict_proba(fisher_test)

	print('Prediction for noise...')

	with tf.Session(graph=noise_model.graph) as sess:
		saver = tf.train.Saver()
		print('   variable initialization ...')
		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()
		file_to_restore = input("\nName of the file to restore (Directory : " + 
                              noise_model.dir_ckpt + ') : ')
		saver.restore(sess, noise_model.dir_ckpt + file_to_restore)

		y_pred_noise = np.empty([nb_batch*batch_size, 2]) 

		for i in range(nb_batch):
			feed_dict = {noise_model.x: data_test[i][0], noise_model.keep_prob: 1.0}
			y_pred_noise[i*batch_size:(i+1)*batch_size] = noise_model.y_conv.eval(feed_dict)


	del(data_test)

	y_pred = np.concatenate([y_pred_noise, y_pred_texture], axis = 1)

	return([y_pred, y_test])

def compute_proba_train(noise_model, texture_model, data, 
				  	   nb_batch, batch_size = 50): 
	
	data_train = []
	for i in range(nb_batch):
		print('Getting batch ' + str(i+1) + '/' + str(nb_batch))
		images_batch, y_batch = data.get_batch_test(batch_size = batch_size,
													crop = False)
		data_train.append([images_batch, y_batch])


	# Texture features 

	features_train_texture = np.empty([nb_batch*batch_size, 128, texture_model.nb_mini_patch])
	y_train = np.empty([nb_batch*batch_size, ])

	pool = Pool()  

	to_compute = [i for i in range(nb_batch)]
	result = pool.starmap(partial(compute_dense_sift, 
								  batch_size = batch_size, 
								  nb_mini_patch = texture_model.nb_mini_patch, 
								  nb_batch = nb_batch,
								  only_green = texture_model.only_green),
								  zip(data_train, to_compute)) 


	
	index = 0
	for i in range(len(result)):
		features_train_texture[index:index+batch_size] = result[i][0]
		y_train[index:index+batch_size] = result[i][1]

		index+=batch_size

	del(result)

	print('Dimension reduction...')
	features_train_PCA = np.empty([nb_batch*batch_size, texture_model.keep_PCA, texture_model.nb_mini_patch])
	for i in range(texture_model.nb_mini_patch):
		# normalize(features_test[:,:,i])
		# features_test_PCA[:,:,i] = pca.transform(features_test[:,:,i])
		features_train_PCA[:,:,i] = texture_model.PCAs[i].transform(features_train_texture[:,:,i])

	del(features_train_texture)

	print('Computing Fisher vectors...')
	fisher_train = compute_fisher(features_train_PCA, texture_model.gmm)

	del(features_train_PCA)

	print('Prediction for textures...')
	y_pred_texture = texture_model.clf_svm.predict_proba(fisher_train)

	print('Prediction for noise...')

	with tf.Session(graph=noise_model.graph) as sess:
		saver = tf.train.Saver()
		print('   variable initialization ...')
		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()
		file_to_restore = input("\nName of the file to restore (Directory : " + 
                              noise_model.dir_ckpt + ') : ')
		saver.restore(sess, noise_model.dir_ckpt + file_to_restore)

		y_pred_noise = np.empty([nb_batch*batch_size, 2]) 

		for i in range(nb_batch):
			feed_dict = {noise_model.x: data_train[i][0], noise_model.keep_prob: 1.0}
			y_pred_noise[i*batch_size, (i+1)*batch_size] = noise_model.y_conv.eval(feed_dict)


	del(data_train)

	y_pred = np.concatenate([y_pred_noise, y_pred_texture], axis = 1)

	return([y_pred, y_train])

config = 'server'

if config == 'server':
	data_directory = '/work/smg/v-nicolas/level-design_raise_100/'
	texture_model_directory = '/work/smg/v-nicolas/models_texture/'
else:
	data_directory = '/home/nicolas/Database/level-design_raise_100_color/'
	texture_model_directory = '/home/nicolas/Documents/models_texture/'

image_size = 100

noise_model = Model(data_directory, image_size, config = 'Server', 
					filters = [32,64], batch_size = 50, 
					feature_extractor = 'Stats', remove_context = False)

texture_model_name = input('Texture model to load : ')

texture_model = load_model(texture_model_directory + texture_model_name + '.pkl')

nb_train_batch = 200
nb_test_batch = 80

[y_pred_train, y_train] = compute_proba_train(noise_model, texture_model,
											  noise_model.data, nb_batch = nb_train_batch)

clf = SVC()

clf.fit(y_pred_train, y_train)

[y_pred_test, y_test] = compute_proba_test(noise_model, texture_model,
											 noise_model.data, nb_batch = nb_test_batch)

y_pred_final = clf.predict(y_pred_test)

score = accuracy_score(y_pred_final, y_test)

print('Final accuracy : ' + str(score))
