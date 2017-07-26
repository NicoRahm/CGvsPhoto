import numpy as np 
from CGvsPhoto import image_loader as il

from multiprocessing import Pool

from functools import partial

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

import cv2

def compute_code(minipatch, mode = 'ltc'): 

	s = np.sign(minipatch - minipatch[1,1])
	# print(s)
	if mode == 'lbp':
		s[s == -1] = 0
		# print(s)
		binary = array_to_bin(s)
		# print(binary)

		return(binary)
	if mode == 'ltc': 
		return((str(int(np.sum(s[s==1])))+ 'u', str(-int(np.sum(s[s == -1]))) + 'l'))
	# return('0b100000000')

def get_classes(mode = 'ltc'):
	classes = dict()
	if mode == 'lbp':
		n = 0
		for i in range(511):
			b = '{:08b}'.format(i)
			A = [[int(b[0]), int(b[1]), int(b[2])], 
				 [int(b[7]), 0, int(b[3])],
				 [int(b[6]), int(b[5]), int(b[4])]]

			b = array_to_bin(np.array(A))
			if b not in classes:
				classes[b] = n
				n+=1 
		
	if mode == 'ltc': 
		n = 0
		for i in range(9):
			classes[str(i) + 'l'] = n
			classes[str(i) + 'u'] = n + 1
			n+=2

	print(str(n) + ' classes')
	return(classes)

def compute_error_image(image): 
	prediction = np.empty([image.shape[0] - 1, image.shape[1] - 1])
	for i in range(image.shape[0]-1):
		for j in range(image.shape[1]-1):
			a = image[i, j+1] 
			b = image[i+1, j]
			c = image[i+1, j+1]

			if c <= min(a,b):
				prediction[i,j] = max(a,b)
			else: 
				if c >= max(a,b):
					prediction[i,j] = min(a,b)
				else: 
					prediction[i,j] = a + b -c
	# print(prediction.shape)
	# print(image.shape)
	error = image[:image.shape[0]-1, :image.shape[1]-1, 0] - prediction 
	# print(error.shape)
	return(error)

def array_to_bin(A): 

	T = [int(A[0,0]), int(A[0,1]), int(A[0,2]), int(A[1,2]), int(A[2,2]),
	     int(A[2,1]), int(A[2,0]), int(A[1,0])]

	nb_c = 0
	for i in range(1,8): 
		if T[i-1] != T[i]: 
			nb_c += 1

	if nb_c > 2: 
		binary = '0b100000000'
	else: 
		binary = '0b' + str(T[0]) + str(T[1]) + str(T[2]) + str(T[3]) + str(T[4]) + str(T[5]) + str(T[6]) + str(T[7])

	return(binary)

def compute_jpeg_coef(image): 

	height = image.shape[1]
	width = image.shape[0]
	nb_channels = image.shape[2]
	result = np.zeros([8*int(width/8), 8*int(height/8), nb_channels], dtype = np.float32)

	for c in range(nb_channels):
		for i in range(int(width/8)):
			for j in range(int(height/8)): 
				result[8*i:8*(i+1), 8*j:8*(j+1), c] = np.round(cv2.dct(np.float32((image[8*i:8*(i+1), 8*j:8*(j+1), c])-128)))
				# print(np.max(result[8*i:8*(i+1), 8*j:8*(j+1), c]))
	return(result)

def compute_hist(image, mode = 'ltc'): 

	hist_1 = dict()
	hist_2 = dict()
	# hist_error = dict()
	for i in classes.keys():
		hist_1[i] = 0
		hist_2[i] = 0
		# hist_error[i] = 0

	image = cv2.cvtColor(image*255, cv2.COLOR_RGB2YCR_CB)
	# image = compute_jpeg_coef(image)
	# error = compute_error_image(image)

	for i in range(1, image.shape[0] - 2): 
		for j in range(1, image.shape[1] - 2): 
			if mode == 'lbp':
				b = compute_code(image[i-1:i+2, j-1:j+2,0], mode)
				hist_1[b] += 1
				# b = compute_code(image[i-1:i+2, j-1:j+2,1], mode)
				# hist_2[b] += 1

			if mode == 'ltc':
				b = compute_code(image[i-1:i+2, j-1:j+2,0], mode)
				hist_1[b[0]] += 1
				hist_1[b[1]] += 1
				# b = compute_code(image[i-1:i+2, j-1:j+2,1], mode)
				# hist_2[b[0]] += 1
				# hist_2[b[1]] += 1				
			# b_error = compute_code(error[i-1:i+2, j-1:j+2])
			# hist_error[b_error] += 1

	F = []
	N = (image.shape[0] - 3)*(image.shape[1] - 3)
	for i in hist_1.keys():
		F.append(hist_1[i]/N)
		# F.append(hist_2[i]/N)
		# F.append(hist_error[i])

	return(np.array(F))


def compute_features(data, i, batch_size, nb_batch, mode = 'ltc'): 

	print('Compute features for batch ' + str(i+1) + '/' + str(nb_batch))
	images, labels = data[0], data[1]
	features = []
	y_train = []
	for i in range(batch_size): 
		features.append(compute_hist(images, mode))
		y_train.append(labels[0])

	# print(features[0])
	return(features, y_train)

def compute_testing_features(i, batch_size, nb_test_batch, data): 

	print('Compute features for testing batch ' + str(i+1) + '/' + str(nb_test_batch))
	images, labels = data.get_batch_test(batch_size = batch_size,
											   crop = False)

	features = []
	y_test = []
	for i in range(batch_size): 
		features.append(compute_hist(images))
		y_test.append(labels[0])

	return(features, y_test)

if __name__ == '__main__': 

	data_directory = '/work/smg/v-nicolas/level-design_raise_100_color/'
	image_size = None

	data = il.Database_loader(directory = data_directory, 
							  size = image_size, only_green = False)

	mode = 'lbp'

	classes = get_classes(mode)

	nb_train_batch = 400
	batch_size = 32

	print('Training...')
	features_train = np.empty([nb_train_batch*batch_size, len(classes.keys())])
	y_train = np.empty([nb_train_batch*batch_size,])

	pool = Pool()  
	index = 0
	for i in range(nb_train_batch):
		data_train = []
		print('Getting batch ' + str(i+1) + '/' + str(nb_train_batch))
		for j in range(batch_size):
			print('Getting image ' + str(j+1) + '/' + str(batch_size))
			images_batch, y_batch = data.get_next_train(crop = False)
			data_train.append([images_batch, y_batch])

	

		to_compute = [i for i in range(batch_size)]
		result = pool.starmap(partial(compute_features, 
								  batch_size = 1, 
								  nb_batch = batch_size, 
								  mode = mode),
								  zip(data_train, to_compute)) 


	

		
		for i in range(len(result)):
			features_train[index:index+1] = result[i][0]
			y_train[index:index+1] = result[i][1]

			index+=1

	del(data_train)
	del(result)

	features_train = normalize(features_train, axis = 1)
	print(features_train[0], y_train[0])
	print(features_train[1], y_train[1])
	print(features_train[2], y_train[2])

	clf = SVC()


	print('Fitting SVM...')

	clf.fit(features_train, y_train)

	y_pred = clf.predict(features_train)

	score = accuracy_score(y_pred,y_train)

	print("Accuracy : " + str(score))

	print('Testing...')

	nb_test_batch = 100

	features_test = np.empty([nb_test_batch*batch_size, len(classes.keys())])
	y_test = np.empty([nb_test_batch*batch_size,])



	data_test = []
	index = 0
	for i in range(nb_test_batch):
		print('Getting batch ' + str(i+1) + '/' + str(nb_test_batch))
		images_batch, y_batch = data.get_next_test(crop = False)
		data_test.append([images_batch, y_batch])
		for j in range(batch_size):
			print('Getting image ' + str(j+1) + '/' + str(batch_size))
			images_batch, y_batch = data.get_next_train(crop = False)
			data_test.append([images_batch, y_batch])

	

		to_compute = [i for i in range(batch_size)]
		result = pool.starmap(partial(compute_features, 
								  batch_size = 1, 
								  nb_batch = batch_size, 
								  mode = mode),
								  zip(data_test, to_compute)) 
		for i in range(len(result)):
			features_test[index:index+1] = result[i][0]
			y_test[index:index+1] = result[i][1]

			index+=1


	del(data_test)



	del(result)

	features_test = normalize(features_test, axis = 1)
	print(features_test[0], y_test[0])
	print(features_test[1], y_test[1])
	print(features_test[2], y_test[2])

	print('Prediction...')
	y_pred = clf.predict(features_test)

	score = accuracy_score(y_pred,y_test)

	print("Accuracy : " + str(score))


