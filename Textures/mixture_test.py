from CGvsPhoto import Model
from CGvsPhoto import image_loader as il
from texture import Texture_model, load_model, compute_dense_sift, compute_fisher

from multiprocessing import Pool

from functools import partial

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import tensorflow as tf

import os 


def compute_features_noise(data, noise_model, noise_model_name):
	'''
	Computes noise features from
	data format : [] tab with size nb_batch containing numpy array containing batches
	'''

	nb_batch = len(data)
	batch_size = data[0].shape[0]
	with tf.Session(graph=noise_model.graph) as sess:
		saver = tf.train.Saver()
		print('   variable initialization ...')
		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()

		saver.restore(sess, noise_model.dir_ckpt + noise_model_name)

		y_pred_noise = np.empty([nb_batch*batch_size, 2]) 

		for i in range(nb_batch):
			feed_dict = {noise_model.x: data[i], noise_model.keep_prob: 1.0}
			y_pred_noise[i*batch_size:(i+1)*batch_size] = noise_model.y_conv.eval(feed_dict)

	return(y_pred_noise)


def compute_features_texture(data, texture_model): 

	# Texture features 
	nb_batch = len(data)
	batch_size = data[0].shape[0]

	features_test_texture = np.empty([nb_batch*batch_size, 128, texture_model.nb_mini_patch])

	pool = Pool()  

	to_compute = [i for i in range(nb_batch)]
	result = pool.starmap(partial(compute_dense_sift, 
								  batch_size = batch_size, 
								  nb_mini_patch = texture_model.nb_mini_patch, 
								  nb_batch = nb_batch,
								  only_green = texture_model.only_green),
								  zip(data, to_compute)) 


	
	index = 0
	for i in range(len(result)):
		features_test_texture[index:index+batch_size] = result[i][0]

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

	return(y_pred_texture)


def compute_proba_test(noise_model, noise_model_name, texture_model,
					   data_loader, nb_batch, batch_size = 50): 
	
	data_test = []
	y_test_batch = []
	for i in range(nb_batch):
		print('Getting batch ' + str(i+1) + '/' + str(nb_batch))
		images_batch, y_batch = data_loader.get_batch_test(batch_size = batch_size,
													crop = False)
		data_test.append(images_batch)
		y_test_batch.append(y_batch)

	y_test = np.empty([nb_batch*batch_size, ])
	index = 0
	for i in range(len(data_test)):
		y_test[index:index+batch_size] = y_test_batch[i][:,0]

		index+=batch_size

	y_pred_noise = compute_features_noise(data_test, noise_model, noise_model_name)
	y_pred_texture = compute_features_texture(data_test, texture_model)

	del(data_test)

	y_pred = np.concatenate([y_pred_noise, y_pred_texture], axis = 1)

	return([y_pred, y_test])

def compute_proba_train(noise_model, noise_model_name, texture_model, 
					   data_loader, nb_batch, batch_size = 50): 
	
	data_train = []
	y_train_batch = []
	for i in range(nb_batch):
		print('Getting batch ' + str(i+1) + '/' + str(nb_batch))
		images_batch, y_batch = data_loader.get_next_train_batch(batch_size = batch_size,
														  crop = False)
		data_train.append(images_batch)
		y_train_batch.append(y_batch)

	y_train = np.empty([nb_batch*batch_size, ])

	index = 0
	for i in range(len(data_train)):
		y_train[index:index+batch_size] = y_train_batch[i][:,0]

		index+=batch_size

	y_pred_noise = compute_features_noise(data_train, noise_model, noise_model_name)
	y_pred_texture = compute_features_texture(data_train, texture_model)

	del(data_train)

	y_pred = np.concatenate([y_pred_noise, y_pred_texture], axis = 1)

	return([y_pred, y_train])


def test_total_images(test_data_path, nb_images, noise_model, 
					  noise_model_name, texture_model, mixture_clf, 
					  minibatch_size = 50, show_images = False, 
					  save_images = False, only_green = True): 

	test_name = input("   Choose a name for the test : ")
	if(save_images):
		if not os.path.exists(noise_model.dir_visualization + test_name):
			os.mkdir(noise_model.dir_visualization + test_name)

	if not only_green:
		print('   No visualization when testing all channels...')
		show_images = False
		save_images = False
	print('   Testing for the database : ' + test_data_path)

	data_test = il.Test_loader(test_data_path, subimage_size = noise_model.image_size, only_green = only_green)


	tp = 0
	fp = 0
	nb_CGG = 0
	accuracy = 0
	for i in range(nb_images):
		batch, label, width, height, original = data_test.get_next_image()
		if not only_green: 
			batch = np.reshape(batch, (batch.shape[0]*3, batch.shape[1], batch.shape[2],1))

		patches = []
		for p in range(int(batch.shape[0]/minibatch_size) + 1):
			patches.append(batch[p:p+minibatch_size])


		y_pred_texture = compute_features_texture(patches, texture_model)
		y_pred_noise = compute_features_noise(patches, noise_model, noise_model_name)
		y_pred = np.concatenate([y_pred_noise, y_pred_texture], axis = 1)
		
		final_pred = np.log(0.000001 + mixture_clf.predict_proba(y_pred))

		prediction = 0
		labels = []
		diff = []
		label_image = np.argmax(final_pred, 1)
		d =  np.max(final_pred, 1) - np.min(final_pred, 1)
		for k in range(d.shape[0]):
			diff.append(np.round(d[k], 1))

		prediction += np.sum(2*d*(label_image - 0.5))

		for l in label_image:
			labels.append(data_test.image_class[l])

		diff = np.array(diff)
		prediction = data_test.image_class[int(max(prediction,0)/abs(prediction))]
		if label == 'CGG':
			nb_CGG += 1

		if(label == prediction):
			accuracy+= 1
			if(prediction == 'CGG'):
				tp += 1
		else:
			if prediction == 'CGG':
				fp += 1
		print(prediction, label)

		if show_images and not save_images:
			test_name = ''

		if save_images or show_images:
			noise_model.image_visualization(path_save = noise_model.dir_visualization + test_name, 
                                   	file_name = str(i), 
                                   	images = batch, labels_pred = labels, 
                                   	true_label = label, width = width, 
                                   	height = height, diff = diff,
                                   	original = original,
                                   	show_images = show_images,
                                   	save_images = save_images,
                                   	save_original = save_images,
                                   	prob_map = save_images)

		if ((i+1)%10 == 0):
			print('\n_______________________________________________________')
			print(str(i+1) + '/' + str(nb_images) + ' images treated.')
			print('Accuracy : ' + str(round(100*accuracy/(i+1), 2)) + '%')
			if tp + fp != 0:
				print('Precision : ' + str(round(100*tp/(tp + fp), 2)) + '%')
			if nb_CGG != 0:
				print('Recall : ' + str(round(100*tp/nb_CGG,2)) + '%')
			print('_______________________________________________________\n')









config = 'server'

if config == 'server':
	data_directory = '/work/smg/v-nicolas/level-design_raise_100/'
	texture_model_directory = '/work/smg/v-nicolas/models_texture/'
	test_data_path = '/work/smg/v-nicolas/level-design_raise/test/'
else:
	data_directory = '/home/nicolas/Database/level-design_raise_100_color/'
	texture_model_directory = '/home/nicolas/Documents/models_texture/'
	test_data_path = '/home/nicolas/Database/level-design_raise/test/'

image_size = 100

noise_model = Model(data_directory, image_size, config = 'Server', 
					filters = [32,64], batch_size = 50, 
					feature_extractor = 'Stats', remove_context = False)

texture_model_name = input('Texture model to load : ')

texture_model = load_model(texture_model_directory + texture_model_name + '.pkl')

noise_model_name = input("\nName of the file to restore (Directory : " + 
                        noise_model.dir_ckpt + ') : ')

print('Trying to load noise model...')
with tf.Session(graph=noise_model.graph) as sess:
	saver = tf.train.Saver()
	print('   variable initialization ...')
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	saver.restore(sess, noise_model.dir_ckpt + noise_model_name)

print('Successfully loaded.')

nb_train_batch = 200
nb_test_batch = 80

print('Training...')

[y_pred_train, y_train] = compute_proba_train(noise_model, noise_model_name, texture_model,
											  noise_model.data, nb_batch = nb_train_batch)

clf = SVC()

print('Fitting mixture SVM...')
clf.fit(y_pred_train, y_train)


print('\nTesting...')

[y_pred_test, y_test] = compute_proba_test(noise_model, noise_model_name, texture_model,
											 noise_model.data, nb_batch = nb_test_batch)

y_pred_final = clf.predict(y_pred_test)

score = accuracy_score(y_pred_final, y_test)

print('Final accuracy : ' + str(score))


nb_images = 720
test_total_images(test_data_path, nb_images, noise_model, noise_model_name,
				  texture_model, clf, minibatch_size = 50, show_images = False,
				  save_images = False)

