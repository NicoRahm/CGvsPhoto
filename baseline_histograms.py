print("   reset python interpreter ...")
import os
clear = lambda: os.system('clear')
clear()
import time
import random
# import plot_history as ph
import image_loader as il
import tensorflow as tf
# import matplotlib.pyplot as plt

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score as acc

config = ''
# config = 'server'

# tool functions
print('   python function setup')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def extract_features_hist(h): 

  nbins = h.shape[2]
  nb_filters = h.shape[0]
  batch_size = h.shape[1]
  h = np.transpose(h, (1,0,2))
  feat_size = nb_filters*(int(nbins/2) + 1)
  features = np.zeros((batch_size, nb_filters, int(nbins/2) + 1))
  for k in range(int(nbins/2)):
    features[:, :, k] = (h[:, :, k] + h[:, :, nbins-k-1])/2

  features[:, :, int(nbins/2)] = h[:, :, int(nbins/2)]
  features = np.reshape(features, (-1, feat_size))
  # print(features.shape)
  # print(features[1:10])
  # print('Rank : ' + str(np.linalg.matrix_rank(features)))
  return(features)

def train_classifier(database_path, image_size, nb_train_batch,
                     nb_test_batch, batch_size = 50, clf = None):

  # computation time tick
  start_clock = time.clock()
  start_time = time.time()

  # seed initialisation
  print("\n   random initialisation ...")
  random_seed = int(time.time() % 10000 ) 
  random.seed(random_seed)  # for reproducibility
  print('   random seed =', random_seed)

  # start process
  print('   tensorFlow version: ', tf.__version__)

  # load data
  print('   import data : image_size = ' + str(image_size) + 'x' + 
        str(image_size) + '...')
  data = il.Database_loader(database_path, image_size, 
                            proportion = 1, only_green=True)


  print('   create model ...')

  # input layer. One entry is a float size x size, 3-channels image. 
  # None means that the number of such vector can be of any lenght.
  with tf.name_scope('Input_Data'):
    x = tf.placeholder(tf.float32, [None, None, None, 1])

    # reshape the input data:
    # size,size: width and height
    # 1: color channels
    # -1 :  ???
    x_image = x
    with tf.name_scope('Image_Visualization'):
      tf.summary.image('Input_Data', x)

    x_shape = tf.placeholder(tf.float32, [2])
  # Filtering to obtain derivation 
  horizontal = tf.constant([[1,-1],[0,0]], tf.float32)
  horizontal_filter = tf.reshape(horizontal, [2, 2, 1, 1])

  vertical = tf.constant([[1,0],[-1,0]], tf.float32)
  vertical_filter = tf.reshape(vertical, [2, 2, 1, 1])

  diagonal = tf.constant([[1,0], [0,-1]], tf.float32)
  diagonal_filter = tf.reshape(diagonal, [2, 2, 1, 1])

  antidiag = tf.constant([[0,1],[-1,0]], tf.float32)
  antidiag_filter = tf.reshape(antidiag, [2, 2, 1, 1])

  x_filtered = []

  # First order filtering
  x_image_h = conv2d(x_image, horizontal_filter)
  x_filtered.append(x_image_h)

  x_image_v = conv2d(x_image, vertical_filter)
  x_filtered.append(x_image_v)

  x_image_d = conv2d(x_image, diagonal_filter)
  x_filtered.append(x_image_d)

  x_image_a = conv2d(x_image, antidiag_filter)
  x_filtered.append(x_image_a)

  # Second order filtering 
  x_image_hv = conv2d(x_image_v, horizontal_filter)
  x_filtered.append(x_image_hv)
  x_image_hh = conv2d(x_image_h, horizontal_filter)
  x_filtered.append(x_image_hh)
  x_image_hd = conv2d(x_image_d, horizontal_filter)
  x_filtered.append(x_image_hd)
  x_image_dd = conv2d(x_image_d, diagonal_filter)
  x_filtered.append(x_image_dd)
  x_image_ha = conv2d(x_image_a, horizontal_filter)
  x_filtered.append(x_image_ha)
  x_image_aa = conv2d(x_image_a, antidiag_filter)
  x_filtered.append(x_image_aa)
  x_image_vd = conv2d(x_image_d, vertical_filter)
  x_filtered.append(x_image_vd)
  x_image_vv = conv2d(x_image_v, vertical_filter)
  x_filtered.append(x_image_vv)
  x_image_va = conv2d(x_image_a, vertical_filter)
  x_filtered.append(x_image_va)
  x_image_da = conv2d(x_image_a, diagonal_filter)
  x_filtered.append(x_image_da)


  nbins = 11

  hist = []
  function_to_map = lambda x: 1000.0 * tf.histogram_fixed_width(x, 
                                                       [-1.,1.], 
                                                       nbins = nbins, 
                                                       dtype=tf.float32)/(x_shape[1]*x_shape[0])

  for x_filt in x_filtered:
  	hist.append(tf.map_fn(function_to_map, x_filt))

  hist = tf.stack(hist)

  # start a session
  print('   start session ...')
  tf.InteractiveSession()

  print('   variable initialization ...')
  tf.global_variables_initializer().run()

  # training the LDA classifier
  if clf == None:
    clf = LinearDiscriminantAnalysis()

  features = []
  labels = []
  for i in range(nb_train_batch):
    if (i%10 == 0):
      print("Computing features for training batch " + str(i) + '/' + str(nb_train_batch))

    batch = data.get_next_train(crop = False)
    input_image = np.array([batch[0]])
    shape = np.array(batch[0].shape[:2]).astype(np.float32)
    feed_dict = {x: input_image, x_shape: shape}
    h = hist.eval(feed_dict = feed_dict)
    features.append(extract_features_hist(h))
    labels.append(np.argmax(np.array([batch[1]]), 1))
  
  features = np.reshape(np.array(features), (batch_size*nb_train_batch, features[0].shape[1])) 
  labels = np.reshape(np.array(labels), (batch_size*nb_train_batch,)) 
  print(features.shape)
  print(labels.shape)
  clf.fit(features, labels)
    

  print('   final test ...')
  test_accuracy = 0
  nb_iterations = nb_test_batch
  data.test_iterator = 0
  for _ in range( nb_iterations ) :
      if (i%10 == 0):
        print("Testing batch " + str(i) + '/' +str(nb_iterations))
      batch_test = data.get_next_train(crop = False)
      input_image = np.array([batch_test[0]])
      shape = np.array(batch_test[0].shape[:2]).astype(np.float32)
      feed_dict = {x: input_image, x_shape : shape}
      h = hist.eval(feed_dict = feed_dict)
      features = extract_features_hist(h)

      y_pred = clf.predict(features)

      test_accuracy += acc(y_pred, np.argmax(np.array([batch_test[1]]), 1))
            
  test_accuracy /= nb_iterations
  print("   test accuracy %g"%test_accuracy)

  # done
  print("   computation time (cpu) :",time.strftime("%H:%M:%S", time.gmtime(time.clock()-start_clock)))
  print("   computation time (real):",time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
  print('   done.')

  return(clf)



if __name__ == '__main__': 

  if config == 'server':
    database_path = '/work/smg/v-nicolas/level-design_raise'
  else:
    database_path = '/home/nicolas/Database/level-design_dresden'
  image_size = None

  clf = train_classifier(database_path = database_path, 
                         image_size = image_size,
                         nb_train_batch = 2520,
                         nb_test_batch = 720,
                         batch_size = 1)