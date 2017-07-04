import sys
sys.path.append('../CGvsPhoto')

import image_loader as il 
from dsift import DsiftExtractor

# import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from multiprocessing import Pool

from functools import partial


def compute_fisher(X, gmm, alpha = 0.5): 

	weights = gmm.weights_
	means = gmm.means_
	covars = gmm.covariances_

	K = weights.shape[0]
	N = X.shape[0]
	F = X.shape[1]
	T = X.shape[2]

	G = np.empty([N, 2*F*K])
	gamma = np.empty([N,K,T])
	for t in range(T):
		gamma[:,:,t] = gmm.predict_proba(X[:,:,t])

	for i in range(K):

		shifted_X = (X - np.reshape(means[i],[1,F,1]))/np.reshape(covars[i], [1,F,1])

		G_mu = np.sum(shifted_X*gamma[:,i:i+1, :]/(T*np.sqrt(weights[i])), axis = 2)
		G_sig = np.sum(gamma[:,i:i+1, :]*(shifted_X**2 - 1)/(T*np.sqrt(2*weights[i])), axis = 2)

		G[:, 2*i*F:2*(i+1)*F] = np.concatenate([G_mu, G_sig], axis = 1)

	# del(G_mu, G_sig, shifted_X, gamma)
	# Power normalization 
	G = np.sign(G)*np.power(np.abs(G), alpha)

	# L2 normalization
	G = G/np.reshape(np.sqrt(np.sum(G**2, axis = 1)), [N,1])

	return(G)

def compute_training_features(i, batch_size, nb_mini_patch, 
							  nb_train_batch):
	
	
	extractor1 = DsiftExtractor(8,16,1)
	extractor2 = DsiftExtractor(16,32,1)

	print('Compute features for training batch ' + str(i+1) + '/' + str(nb_train_batch))
	images, labels = data.get_next_train_batch(batch_size = batch_size,
											   crop = False)
	features = []
	y_train = []
	for j in range(batch_size):
		img = (images[j]*256).astype(np.uint8)
		img = np.dot(img, [0.299, 0.587, 0.114])
		feaArr1,positions = extractor1.process_image(img, verbose = False)
		feaArr2,positions = extractor2.process_image(img, verbose = False)
		features.append(np.concatenate([feaArr1, feaArr2]).reshape([128, nb_mini_patch]))
		y_train.append(labels[j,0])

	return(features, y_train)

def compute_features(data, i, batch_size, nb_mini_patch, nb_batch):

	extractor1 = DsiftExtractor(8,16,1)
	extractor2 = DsiftExtractor(16,32,1)

	print('Compute features for batch ' + str(i+1) + '/' + str(nb_batch))
	images, labels = data[0], data[1]

	features = []
	y_train = []
	for j in range(batch_size):
		img = (images[j]*256).astype(np.uint8)
		img = np.dot(img, [0.299, 0.587, 0.114])
		feaArr1,positions = extractor1.process_image(img, verbose = False)
		feaArr2,positions = extractor2.process_image(img, verbose = False)
		features.append(np.concatenate([feaArr1, feaArr2]).reshape([128, nb_mini_patch]))
		y_train.append(labels[j,0])

	return(features, y_train)


def compute_testing_features(i, batch_size, nb_mini_patch,
							 nb_test_batch):
	
	extractor1 = DsiftExtractor(8,16,1)
	extractor2 = DsiftExtractor(16,32,1)

	print('Compute features for testing batch ' + str(i+1) + '/' + str(nb_test_batch))
	images, labels = data.get_batch_test(batch_size = batch_size,
											   crop = False)
	features = []
	y_train = []
	for j in range(batch_size):
		img = (images[j]*256).astype(np.uint8)
		img = np.dot(img, [0.299, 0.587, 0.114])
		feaArr1,positions = extractor1.process_image(img, verbose = False)
		feaArr2,positions = extractor2.process_image(img, verbose = False)
		features.append(np.concatenate([feaArr1, feaArr2]).reshape([128, nb_mini_patch]))
		y_train.append(labels[j,0])

	return(np.array(features), np.array(y_train))








# data = np.array([[[2,4],[0,0],[2,1],[0,1],[0,0],[0,5]], 
# 				[[3,3],[0,1],[0,1],[1,1],[0,4],[6,2]],
# 				[[4,6],[1,1],[0,1],[2,1],[4,1],[3,1]]])

# print('Fitting Gaussian Mixture Model...')
# K = 2
# gmm = GaussianMixture(n_components=K, covariance_type='diag')
# gmm.fit(np.reshape(data, [data.shape[0]*2, 6]))

# print('Computing Fisher vectors...')

# fisher_vectors = compute_fisher(data, gmm)

# print(fisher_vectors)




if __name__ == '__main__':

	config = 'server'

	if config == 'server':
		data_directory = '/work/smg/v-nicolas/level-design_raise_100_color/'
	else:
		data_directory = '/home/nicolas/Database/level-design_raise_100_color/'
	image_size = 100

	data = il.Database_loader(directory = data_directory, 
							  size = image_size, only_green = False)

	nb_train_batch = 800
	batch_size = 50
	extractor = DsiftExtractor(8,16,1)

	nb_mini_patch = 121 + 25



	PCAs = []

	n_comp = 64
	for i in range(nb_mini_patch):
		PCAs.append(PCA(n_components=n_comp))

	# pca = PCA(n_components=n_comp)

	clf = LinearSVC()




	features = np.empty([nb_train_batch*batch_size, 128, nb_mini_patch])
	y_train = np.empty([nb_train_batch*batch_size, ])
	print('Training...')

	# for i in range(nb_train_batch):

	# 	print('Compute features for training batch ' + str(i+1) + '/' + str(nb_train_batch))
	# 	images, labels = data.get_next_train_batch(batch_size = batch_size,
	# 											   crop = False)

	# 	for j in range(batch_size):
	# 		img = (images[j]*256).astype(np.uint8)
	# 		feaArr,positions = extractor.process_image(img, verbose = False)
	# 		features[i*batch_size + j] = feaArr.reshape([128, nb_mini_patch])
	# 		y_train[i*batch_size + j] = labels[j,0]


	data_train = []
	for i in range(nb_train_batch):
		print('Getting batch ' + str(i+1) + '/' + str(nb_train_batch))
		images_batch, y_batch = data.get_next_train_batch(batch_size = batch_size,
												   		  crop = False)
		data_train.append([images_batch, y_batch])

	pool = Pool()  

	to_compute = [i for i in range(nb_train_batch)]
	result = pool.starmap(partial(compute_features, 
							  batch_size = batch_size, 
							  nb_mini_patch = nb_mini_patch, 
							  nb_batch = nb_train_batch),
							  zip(data_train, to_compute)) 

	del(data_train)

	index = 0
	for i in range(len(result)):
		features[index:index+batch_size] = result[i][0]
		y_train[index:index+batch_size] = result[i][1]

		index+=batch_size


	del(result)
	# print(y_train)

	print('Fitting PCAs...')
	# for i in range(nb_mini_patch):
	# 	# normalize(features[:,:,i])

	for i in range(nb_mini_patch):
		PCAs[i].fit(features[:,:,i])

	# pca.fit(np.concatenate([features[:,:,i] for i in range(nb_mini_patch)]))

	print('Dimension reduction...')
	features_PCA = np.empty([nb_train_batch*batch_size, n_comp, nb_mini_patch])
	for i in range(nb_mini_patch):
		# features_PCA[:,:,i] = pca.transform(features[:,:,i])
		features_PCA[:,:,i] = PCAs[i].transform(features[:,:,i])

	del(features)

	print('Fitting Gaussian Mixture Model...')
	K = 64
	gmm = GaussianMixture(n_components=K, covariance_type='diag')
	gmm.fit(np.reshape(features_PCA, [features_PCA.shape[0]*nb_mini_patch, n_comp]))

	print('Computing Fisher vectors...')
	fisher_train = compute_fisher(features_PCA, gmm)

	del(features_PCA)

	# Plotting boxplot

	# for i in range(fisher_train.shape[1]):
	# 	print('Computing dataframe...')
		
	# 	data_real = fisher_train[y_train == 0, i]
	# 	data_cg = fisher_train[y_train == 1, i]

	# 	print('Plotting boxplot...')
	# 	plt.figure()
	# 	plt.boxplot([data_real, data_cg])
	# 	plt.show()

	print('Fitting SVM...')
	clf.fit(fisher_train, y_train)
	# clf.fit(np.reshape(features_PCA, [nb_train_batch*batch_size, n_comp*nb_mini_patch]), y_train)



	del(fisher_train, y_train)


	print('Testing...')
	nb_test_batch = 80
	features_test = np.empty([nb_test_batch*batch_size, 128, nb_mini_patch])
	y_test = np.empty([nb_test_batch*batch_size, ])

	# for i in range(nb_test_batch):
	# 	print('Compute features for testing batch ' + str(i+1) + '/' + str(nb_test_batch))

	# 	images, labels = data.get_batch_test(batch_size = batch_size,
	# 											   crop = False)

	# 	for j in range(batch_size):
	# 		img = (images[j]*256).astype(np.uint8)
	# 		feaArr,positions = extractor.process_image(img, verbose = False)
	# 		features_test[i*batch_size + j] = feaArr.reshape([128, nb_mini_patch])
	# 		y_test[i*batch_size + j] = labels[j,0]


	data_test = []
	for i in range(nb_test_batch):
		print('Getting batch ' + str(i+1) + '/' + str(nb_test_batch))
		images_batch, y_batch = data.get_batch_test(batch_size = batch_size,
												   	crop = False)
		data_test.append([images_batch, y_batch])

	pool = Pool()  

	to_compute = [i for i in range(nb_test_batch)]
	result = pool.starmap(partial(compute_features, 
							  batch_size = batch_size, 
							  nb_mini_patch = nb_mini_patch, 
							  nb_batch = nb_test_batch),
							  zip(data_test, to_compute)) 


	del(data_test)

	index = 0
	for i in range(len(result)):
		features_test[index:index+batch_size] = result[i][0]
		y_test[index:index+batch_size] = result[i][1]

		index+=batch_size

	del(result)

	print('Dimension reduction...')
	features_test_PCA = np.empty([nb_test_batch*batch_size, n_comp, nb_mini_patch])
	for i in range(nb_mini_patch):
		# normalize(features_test[:,:,i])
		# features_test_PCA[:,:,i] = pca.transform(features_test[:,:,i])
		features_test_PCA[:,:,i] = PCAs[i].transform(features_test[:,:,i])

	del(features_test)

	print('Computing Fisher vectors...')
	fisher_test = compute_fisher(features_test_PCA, gmm)

	del(features_test_PCA)



	print('Prediction...')
	y_pred = clf.predict(fisher_test)
	# y_pred = clf.predict(np.reshape(features_test_PCA, [nb_test_batch*batch_size, n_comp*nb_mini_patch]))

	print('Computing score...')
	score = accuracy_score(y_pred, y_test)

	print('Accuracy : ' + str(score))
