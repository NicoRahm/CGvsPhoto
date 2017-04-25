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


# computation time tick
start_clock = time.clock()
start_time = time.time()

# seed initialisation
print("\n   random initialisation ...")
random_seed = int(time.time() % 10000 ) 
random.seed(random_seed)  # for reproducibility
print('   random seed =', random_seed)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# tool functions
print('   python function setup')
def weight_variable(shape, seed = None):
  initial = tf.truncated_normal(shape, stddev=0.1, seed = seed)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

def max_pool_10x10(x):
  return tf.nn.max_pool(x, ksize=[1, 10, 10, 1],
                           strides=[1, 10, 10, 1], padding='SAME')

def avg_pool_10x10(x):
  return tf.nn.avg_pool(x, ksize=[1, 10, 10, 1],
                           strides=[1, 10, 10, 1], padding='SAME')

  # start process
print('   tensorFlow version: ', tf.__version__)
image_size = 200



# load data
print('   import data : image_size = ' + str(image_size) + 'x' + str(image_size) + '...')
# data = il.Database_loader('/home/nozick/Desktop/database/cg_pi_64/test5', image_size, only_green=True)
data = il.Database_loader('/home/nicolas/Documents/Test_DB_200', image_size, proportion = 1, only_green=True)
# data = il.Database_loader('/home/nicolas/Documents/Test', image_size, proportion = 1, only_green=True)




print('   create model ...')
# input layer. One entry is a float size x size, 3-channels image. 
# None means that the number of such vector can be of any lenght.
with tf.name_scope('Input_Data'):
  x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])

# reshape the input data:
# size,size: width and height
# 1: color channels
# -1 :  ???
  x_image = tf.reshape(x, [-1,image_size, image_size, 1])
  with tf.name_scope('Image_Visualization'):
    tf.summary.image('Input_Data', x_image)
    variable_summaries(x_image)

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


nb_filters = 14
nbins = 15

size_features = nbins*nb_filters 

hist = []
function_to_map = lambda x: tf.histogram_fixed_width(x, [-1.0,1.0], nbins = nbins, dtype=tf.float32)

for x_filt in x_filtered:
	hist.append(tf.map_fn(function_to_map, x_filt))

hist = tf.stack(hist)

hist_features = tf.reshape(tf.transpose(hist, [1,2,0]), [-1,size_features], name = "Histogram_features")
variable_summaries(hist_features)

with tf.variable_scope('Dense1'):
  with tf.name_scope('Weights'):
    W_fc1 = weight_variable([size_features, 1024])
    variable_summaries(W_fc1)
  with tf.name_scope('Bias'):
    b_fc1 = bias_variable([1024])
    variable_summaries(b_fc1)
  # put a relu
  h_fc1 = tf.nn.relu(tf.matmul(hist_features, W_fc1) + b_fc1, name = 'activated')
  tf.summary.histogram('activated', h_fc1)

# dropout
with tf.name_scope('Dropout1'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Densely Connected Layer
# we add a fully-connected layer with 1024 neurons 

# with tf.variable_scope('Dense2'):
#   with tf.name_scope('Weights'):
#     W_fc2 = weight_variable([1024, 1024]) 
#     variable_summaries(W_fc2)
#   with tf.name_scope('Bias'):
#     b_fc2 = bias_variable([1024])
#     variable_summaries(b_fc2)
#   # put a relu
#   h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name = 'activated')
#   tf.summary.histogram('activated', h_fc2)

# # dropout
# with tf.name_scope('Dropout2'):
#   keep_prob2 = tf.placeholder(tf.float32)
#   tf.summary.scalar('dropout_keep_probability', keep_prob)
#   h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# readout layer
with tf.variable_scope('Readout'):
  with tf.name_scope('Weights'):
    W_fc3 = weight_variable([1024, data.nb_class])
    variable_summaries(W_fc3)
  with tf.name_scope('Bias'):
    b_fc3 = bias_variable([data.nb_class])
    variable_summaries(b_fc3)
  y_hist = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

# support for the learning label
y_ = tf.placeholder(tf.float32, [None, data.nb_class])


# Define loss (cost) function and optimizer
print('   setup loss function and optimizer ...')

# softmax to have normalized class probabilities + cross-entropy
with tf.name_scope('cross_entropy'):

  softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_hist)
  #print('\nsoftmax_cross_entropy shape : ', softmax_cross_entropy.get_shape() )
  with tf.name_scope('total'):
    cross_entropy_mean = tf.reduce_mean(softmax_cross_entropy)

tf.summary.scalar('cross_entropy', cross_entropy_mean)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)

print('   test ...')
# 'correct_prediction' is a function. argmax(y, 1), here 1 is for the axis number 1
correct_prediction = tf.equal(tf.argmax(y_hist,1), tf.argmax(y_,1))

# 'accuracy' is a function: cast the boolean prediction to float and average them
with tf.name_scope('accuracy'):
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)



# start a session
print('   start session ...')
sess = tf.InteractiveSession()

merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('/home/smg/v-nicolas/summaries',
#                                       sess.graph)

print('   variable initialization ...')
tf.global_variables_initializer().run()

# print(gaussian_func(0., a, 1, 1.).eval())
# print(classic_histogram_gaussian(a, 1, nbins = 8, values_range = [0, 1], sigma = 0.6).eval())

# Train
# print('   train ...')
# history = []
# for i in range(151): # in the test 20000
  
#     # evry 100 batches, test the accuracy
#     if i%10 == 0 :
#         validation_batch_size = 10       # size of the batches
#         validation_accuracy = 0
#         data.validation_iterator = 0

#         nb_iterations = 25
#         for _ in range( nb_iterations ) :
#             batch_validation = data.get_batch_validation(batch_size=validation_batch_size, random_flip_flop = True, random_rotate = True)
#             feed_dict = {x:batch_validation[0], y_: batch_validation[1], keep_prob: 1.0}
#             validation_accuracy += accuracy.eval(feed_dict)
          
#         validation_accuracy /= nb_iterations
#         print("     step %d, training accuracy %g (%d validations tests)"%(i, validation_accuracy, validation_batch_size*nb_iterations))

#         history.append(validation_accuracy)
        
        
#     # regular training
# #    print('get batch ...')
#     batch_size = 50
#     batch = data.get_next_train_batch(batch_size, True, True)
# #    print('train ...')
#     feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.8}
#     summary, _ = sess.run([merged, train_step], feed_dict = feed_dict)
#     train_writer.add_summary(summary, i)


# history
# print('   plot history')
# with open("/tmp/history.txt", "w") as history_file:
#     for item in history:
#         history_file.write("%f\n" %item)

# with open("./history_v2.txt", "w") as history_file:
#     for item in history:
#         history_file.write("%f\n" %item)
        
# ph.plot_history("/tmp/history.txt")


# training the LDA classifier

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
  features = np.reshape(features, (batch_size, feat_size))
  return(features)


clf = LinearDiscriminantAnalysis()
for i in range(1001):
  batch_size = 100
  batch = data.get_next_train_batch(batch_size, False, True, True)
  
  feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.8}
  h = hist.eval(feed_dict = feed_dict)

  features = extract_features_hist(h)
  
  clf.fit(features, np.transpose(batch[1], (1,0))[0])
  if (i%10 == 0):
    print("Training batch " + str(i))

print('   final test ...')
test_batch_size = 10       # size of the batches
test_accuracy = 0
nb_iterations = 200
data.test_iterator = 0
for _ in range( nb_iterations ) :
    batch_test = data.get_batch_test(batch_size=test_batch_size, crop = False, random_flip_flop = True, random_rotate = True)
    feed_dict = {x:batch_test[0], y_: batch_test[1], keep_prob: 1.0}
    h = hist.eval(feed_dict = feed_dict)
    features = extract_features_hist(h)

    y_pred = clf.predict(features)

    test_accuracy += acc(y_pred, np.transpose(batch_test[1], (1,0))[0])
          
test_accuracy /= nb_iterations
print("   test accuracy %g"%test_accuracy)


# final test
# print('   final test ...')
# test_batch_size = 10       # size of the batches
# test_accuracy = 0
# nb_iterations = 200
# data.test_iterator = 0
# for _ in range( nb_iterations ) :
#     batch_test = data.get_batch_test(batch_size=test_batch_size, random_flip_flop = True, random_rotate = True)
#     feed_dict = {x:batch_test[0], y_: batch_test[1], keep_prob: 1.0}
#     test_accuracy += accuracy.eval(feed_dict)
          
# test_accuracy /= nb_iterations
# print("   test accuracy %g"%test_accuracy)


#batch_test = data.get_batch_test(max_images=50)
#print("   test accuracy %g"%accuracy.eval(feed_dict={x: batch_test[0], 
#                                                     y_: batch_test[1], 
#                                                     keep_prob: 1.0}))




# done
print("   computation time (cpu) :",time.strftime("%H:%M:%S", time.gmtime(time.clock()-start_clock)))
print("   computation time (real):",time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
print('   done.')