
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
  initial = tf.truncated_normal(shape, stddev=0.5, seed = random_seed)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, seed = random_seed)
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

def histogram(x, nbins):
  h = tf.histogram_fixed_width(x, value_range = [-1.0,1.0], 
                               nbins = nbins, dtype = tf.float32)
  # mom = tf.nn.moments(h, axes = [0]) 
  # h = (h-mom[0])/mom[1]
  return(h)

def gaussian_func(mu, x, n, sigma):
  gauss = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
  return(tf.reduce_sum(gauss.pdf(x)/n))


def gaussian_kernel(x, nbins = 8, values_range = [0, 1], sigma = 0.1,image_size = 100):
  mu_list = np.float32(np.linspace(values_range[0], values_range[1], nbins + 1))
  n = np.float32(image_size**2)
  function_to_map = lambda m : gaussian_func(m, x, n, sigma)
  return(tf.map_fn(function_to_map, mu_list))


# def plot_gaussian_kernel(nbins = 8, values_range = [0, 1], sigma = 0.1):

#   r = values_range[1] - values_range[0]
#   mu_list = []
#   for i in range(nbins+1):
#     mu_list.append(values_range[0] + i*r/(nbins+1))

#   range_plot = np.linspace(values_range[0], values_range[1], 100)

#   plt.figure()
#   for mu in mu_list:
#     plt.plot(range_plot, np.exp(-(range_plot-mu)**2/(sigma**2)))
#   plt.title("Gaussian kernels used for estimating the histograms")
#   plt.show()

# plot_gaussian_kernel()
# print(gaussian_kernel(np.array([[0,10],[10, 0.5]]), nbins = 4, values_range = [0,10], sigma = 6))

def learnable_histogram(x, nbins, k, image_size): 

  l = []
  for i in range(nbins):
    b = bias_variable([k])
    l.append(x - b)

  conv1 = tf.abs(tf.transpose(tf.stack(l), perm = [1,2,3,0,4]))

  W = weight_variable([1, 1, 1, k, k], seed = random_seed)

  relu = tf.nn.relu(tf.constant(1.0, shape=[k]) - tf.nn.conv3d(conv1, W, strides=[1, 1, 1, 1, 1], padding='SAME'))

  avg_pool = tf.reshape(tf.nn.pool(relu, window_shape=[image_size, image_size, 1], 
                        pooling_type = 'AVG', strides=[image_size, image_size, 1], 
                        padding='SAME'), [-1, nbins, k])
  return(avg_pool)

def classic_histogram(x, nbins, k, image_size):

  l = []
  for i in range(nbins):
    b = float(i)/float(nbins)
    l.append(b)

  b = tf.constant(l, shape = [k,nbins])
  x_3d = tf.reshape(x, [-1,image_size, image_size, k, 1])
  W1 = tf.constant(1.0, shape=[1,1,k,1,nbins])
  conv1 = tf.abs(tf.nn.conv3d(x_3d, W1,strides=[1, 1, 1, 1, 1], padding='SAME') - b)
  conv1 = tf.transpose(conv1, [0,1,2,4,3])

  W = tf.constant(1.0/float(nbins), shape=[1,1,nbins,k,k])

  relu = tf.nn.relu(tf.constant(1.0, shape=[k]) - tf.nn.conv3d(conv1, W, strides=[1, 1, 1, 1, 1], padding='SAME'))

  avg_pool = tf.reshape(tf.nn.pool(relu, window_shape=[image_size, image_size, 1], 
                        pooling_type = 'AVG', strides=[image_size, image_size, 1], 
                        padding='SAME'), [-1, nbins, k])
  return(avg_pool)  

def classic_histogram_gaussian(x, k, nbins = 8, values_range = [0, 1], sigma = 0.6):
  function_to_map = lambda y: tf.stack([gaussian_kernel(y[:,:,i], nbins, values_range, sigma) for i in range(k)])
  res = tf.map_fn(function_to_map, x)
  return(res)

  
  # start process
print('   tensorFlow version: ', tf.__version__)
image_size = 100



# load data
print('   import data : image_size = ' + str(image_size) + 'x' + str(image_size) + '...')
# data = il.Database_loader('/home/nozick/Desktop/database/cg_pi_64/test5', image_size, only_green=True)
# data = il.Database_loader('/media/nicolas/Home/nicolas/Documents/Stage 3A/Test', image_size, only_green=True)
data = il.Database_loader('/work/smg/v-nicolas/Test_DB_100', image_size, proportion = 1, only_green=True)


folder_ckpt = '/work/smg/v-nicolas/weights/'
filename = input("Choose a file name for the weigths : ")

path_save = folder_ckpt + filename


print('   create model ...')
# input layer. One entry is a float size x size, 3-channels image. 
# None means that the number of such vector can be of any lenght.

graph = tf.Graph()

with graph.as_default():

  with tf.name_scope('Input_Data'):
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])

  # reshape the input data:
  # size,size: width and height
  # 1: color channels
  # -1 :  ???
    x_image = tf.reshape(x, [-1,image_size, image_size, 1])
    with tf.name_scope('Image_Visualization'):
      tf.summary.image('Input_Data', x_image)

  # Filtering with laplacian filter
  # laplacian = tf.constant([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], tf.float32)
  # laplacian_filter = tf.reshape(laplacian, [3, 3, 1, 1])

  # horizontal = tf.constant([[1,-1],[0,0]], tf.float32)
  # horizontal_filter = tf.reshape(horizontal, [2, 2, 1, 1])

  # vertical = tf.constant([[1,0],[-1,0]], tf.float32)
  # vertical_filter = tf.reshape(vertical, [2, 2, 1, 1])

  # diagonal = tf.constant([[1,0], [0,-1]], tf.float32)
  # diagonal_filter = tf.reshape(diagonal, [2, 2, 1, 1])

  # antidiag = tf.constant([[0,1],[-1,0]], tf.float32)
  # antidiag_filter = tf.reshape(antidiag, [2, 2, 1, 1])

  #x_image_filtered = conv2d(x_image, laplacian_filter)
  #x_image_filtered = x_image

  #x_image_h = conv2d(x_image, horizontal_filter)
  #x_image_v = conv2d(x_image, vertical_filter)
  #x_image_d = conv2d(x_image, diagonal_filter)
  #x_image_a = conv2d(x_image, antidiag_filter)

  #hist = tf.histogram_fixed_width(x_image, [-1.0,1.0], nbins = 100, dtype=tf.float32)

  # first conv net layer
  # conv_matrix_width : 5
  # conv_matrix_height : 5
  # conv_matrix nb channel ??? : 1
  # nb matrices : 32 ?

  nb_conv1 = 32
  filter_size1 = 3

  with tf.name_scope('Conv1'):

    with tf.name_scope('Weights'):
      W_conv1 = weight_variable([filter_size1, filter_size1, 1, nb_conv1], seed = random_seed)
      # variable_summaries(W_conv1)
    # with tf.name_scope('Bias'):
      # b_conv1 = bias_variable([nb_conv1])
      # variable_summaries(b_conv1)

    # relu on the conv layer
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1), name = 'activated')
    # tf.summary.histogram('activated', h_conv1)

    # with tf.variable_scope('Conv1_visualization'):
    #   tf.summary.image('conv1/filters', W_conv1[:,:,:,0:1])



  # with tf.name_scope('MaxPool'):
  #   m_pool = max_pool_2x2(h_conv1)


  # second conv 

  nb_conv2 = 64
  filter_size2 = 3
  with tf.name_scope('Conv2'):
    with tf.name_scope('Weights'):
      W_conv2 = weight_variable([filter_size2, filter_size2, nb_conv1, nb_conv2])
      variable_summaries(W_conv1)
    # with tf.name_scope('Bias'):
    #   b_conv2 = bias_variable([nb_conv2])
    #   variable_summaries(b_conv2)

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2))
    # tf.summary.histogram('activated', h_conv2)

    with tf.variable_scope('Conv2_visualization'):
      tf.summary.image('conv2/filters', W_conv2[:,:,:,0:1])


  # size_pool = int(nb_conv1*(image_size**2)/4)
  # with tf.name_scope('MaxPool'):
  #   m_pool = max_pool_2x2(h_conv1)



  nbins = 10
  nb_filters = nb_conv2
  size_hist = (nbins + 1)*nb_filters

  # with tf.name_scope('Histograms'):
  #   function_to_map = lambda x: tf.stack([histogram(x[:,:,i], nbins) for i in range(32)])
  #   hist = tf.map_fn(function_to_map, h_conv1)
  #   variable_summaries(hist)


  # with tf.name_scope('Classic_Histogram'): 
  #   hist = classic_histogram(h_conv1, nbins = nbins, k = nb_conv1, image_size = image_size)

  range_hist = [0,1]
  sigma = 0.07

  # plot_gaussian_kernel(nbins = nbins, values_range = range_hist, sigma = 0.1)


  with tf.name_scope('Gaussian_Histogram'): 
    hist = classic_histogram_gaussian(h_conv2, k = nb_filters, nbins = nbins, values_range = range_hist, sigma = sigma)
    tf.summary.tensor_summary('hist', hist)


  # max pool
  #m_pool3 = max_pool_10x10(h_conv1)
  # m_pool3 = max_pool_2x2(h_conv2)

  # average pool
  # a_pool3 = avg_pool_2x2(h_conv2)
  #m_pool4 = max_pool_2x2(h_conv2)

  # difference between max and average
  # diff_1 = tf.subtract(m_pool3, a_pool3)
  # diff_1 = m_pool3

  # flattern 
  # with the 2 pooling, the image size is 7x7
  # there are 64 conv matrices

  # size_conv = tf.cast((image_size/2)*(image_size/2)*32, tf.int32)
  # h_pool1_flat = tf.reshape(m_pool, [-1, size_conv], name = "Flatten_Conv")

  h_pool2_flat = tf.reshape(hist, [-1, size_hist], name = "Flatten_Hist")


  # concat = tf.concat([h_pool1_flat, h_pool2_flat], 1, name = "Concatenate")

  # Densely Connected Layer
  # we add a fully-connected layer with 1024 neurons 

  with tf.variable_scope('Dense1'):
    with tf.name_scope('Weights'):
      W_fc1 = weight_variable([size_hist, 1024])
      # variable_summaries(W_fc1)
    with tf.name_scope('Bias'):
      b_fc1 = bias_variable([1024])
      # variable_summaries(b_fc1)
    # put a relu
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name = 'activated')
    # tf.summary.histogram('activated', h_fc1)

  # dropout
  with tf.name_scope('Dropout1'):
    keep_prob = tf.placeholder(tf.float32)
    # tf.summary.scalar('dropout_keep_probability', keep_prob)
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
  #   # keep_prob2 = tf.placeholder(tf.float32)
  #   # tf.summary.scalar('dropout_keep_probability', keep_prob)
  #   h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  # readout layer
  with tf.variable_scope('Readout'):
    with tf.name_scope('Weights'):
      W_fc3 = weight_variable([1024, data.nb_class])
      # variable_summaries(W_fc3)
    with tf.name_scope('Bias'):
      b_fc3 = bias_variable([data.nb_class])
      # variable_summaries(b_fc3)
    y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

  # support for the learning label
  y_ = tf.placeholder(tf.float32, [None, data.nb_class])




  # Define loss (cost) function and optimizer
  print('   setup loss function and optimizer ...')

  # softmax to have normalized class probabilities + cross-entropy
  with tf.name_scope('cross_entropy'):

    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
    #print('\nsoftmax_cross_entropy shape : ', softmax_cross_entropy.get_shape() )
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(softmax_cross_entropy)

  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy_mean)

  print('   test ...')
  # 'correct_prediction' is a function. argmax(y, 1), here 1 is for the axis number 1
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

  # 'accuracy' is a function: cast the boolean prediction to float and average them
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.name_scope('AUC'):
    auc = tf.metrics.auc(y_, y_conv)

  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('AUC', auc)


# a = tf.constant([[[[1.0,1.0],[1.0,1.]]]])


# start a session
print('   start session ...')
with tf.Session(graph=graph) as sess:

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('/home/smg/v-nicolas/summaries',
                                        sess.graph)

  saver = tf.train.Saver()
  print('   variable initialization ...')

  restore_weigths = input("Restore weight from previous session ? (y/N) : ")

  if restore_weigths == 'y':
    file_to_restore = input("Name of the file to restore (Directory : " + folder_ckpt + ') : ')
    saver.restore(sess, folder_ckpt + file_to_restore)
    print('   Model restored')

  else: 
    tf.global_variables_initializer().run()
    

  

  # print(gaussian_func(0., a, 1, 1.).eval())
  # print(classic_histogram_gaussian(a, 1, nbins = 8, values_range = [0, 1], sigma = 0.6).eval())

  # Train
  print('   train ...')
  history = []
  for i in range(0): # in the test 20000
    
      # evry 100 batches, test the accuracy
      if i%10 == 0 :
          validation_batch_size = 20       # size of the batches
          validation_accuracy = 0
          validation_auc = 0
          data.validation_iterator = 0
          nb_iterations = 50

          selected_hist_nb = 4
          nb_CGG = 0
          hist_CGG = [np.zeros((nbins+1,)) for i in range(selected_hist_nb)]
          nb_real = 0
          hist_real = [np.zeros((nbins+1,)) for i in range(selected_hist_nb)]
          for _ in range( nb_iterations ) :
              batch_validation = data.get_batch_validation(batch_size=validation_batch_size, crop = False, random_flip_flop = True, random_rotate = True)
              feed_dict = {x:batch_validation[0], y_: batch_validation[1], keep_prob: 1.0}
              validation_accuracy += accuracy.eval(feed_dict)
              validation_auc += auc.eval(feed_dict)

          #     # Computing the mean histogram for each class
          #     hist_plot = hist.eval(feed_dict)
          #     for k in range(validation_batch_size): 
          #       if batch_validation[1][k][0] == 0.:
          #         nb_real +=1
          #         is_real = True
          #       else:
          #         nb_CGG += 1
          #         is_real = False
          #       for j in range(selected_hist_nb):
          #         for l in range(nbins+1): 
          #           if is_real:
          #             hist_real[j][l] += hist_plot[k,j,l]
          #           else:
          #             hist_CGG[j][l] += hist_plot[k,j,l]
          
          # for p in range(selected_hist_nb):
          #   hist_CGG[p] /= nb_CGG
          #   hist_real[p] /= nb_real

          # # Plotting mean histogram for CGG
          # fig = plt.figure(1)
          # for k in range(selected_hist_nb):
          #   plt.subplot(selected_hist_nb/2, 2, k+1)
          #   plt.bar(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_CGG[k], width = 1/(nbins + 1))
          #   plt.plot(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_CGG[k], 'r')
          #   fig.suptitle("Mean histogram for CGG", fontsize=14)
          # plt.show()

          # # Plotting mean histogram for Real
          # fig = plt.figure(2)
          # for k in range(selected_hist_nb):
          #   plt.subplot(selected_hist_nb/2, 2, k+1)
          #   plt.bar(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_real[k], width = 1/(nbins + 1))
          #   plt.plot(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_real[k], 'r')
          #   fig.suptitle("Mean histogram for Real", fontsize=14)
          # plt.show()
            
          validation_accuracy /= nb_iterations
          validation_auc /= nb_iterations
          print("     step %d, training accuracy %g (%d validations tests)"%(i, validation_accuracy, validation_batch_size*nb_iterations))
          print("     step %d, training AUC %g (%d validations tests)"%(i, validation_auc, validation_batch_size*nb_iterations))

          # # Plot all histograms for last batch
          # hist_plot = hist.eval(feed_dict)
          
          # for k in range(validation_batch_size):
          #   fig = plt.figure(k)
          #   for j in range(4):
              
          #     plt.subplot(2, 2, j+1)
          #     plt.bar(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_plot[k,j], width = 1/(nbins + 1))
          #     plt.plot(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_plot[k,j], 'r')
            
          #   if batch_validation[1][k][0] == 0.:
          #     fig.suptitle("CGG", fontsize=14)
          #   else:
          #     fig.suptitle("Real", fontsize=14)

          #   plt.show()


          history.append(validation_accuracy)
          
          
      # regular training
  #    print('get batch ...')
      batch_size = 50
      batch = data.get_next_train_batch(batch_size, False, True, True)
  #    print('train ...')
      feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.85}
      summary, _ = sess.run([merged, train_step], feed_dict = feed_dict)
      train_writer.add_summary(summary, i)
      if i%100 == 0:
        
        path_save_batch = path_save + str(i) + ".ckpt"
        print('   saving weights in file : ' + path_save)
        saver.save(sess, path_save_batch)



  # history
  # print('   plot history')
  # with open("/tmp/history.txt", "w") as history_file:
  #     for item in history:
  #         history_file.write("%f\n" %item)

  # with open("./history_v2.txt", "w") as history_file:
  #     for item in history:
  #         history_file.write("%f\n" %item)
          
  # ph.plot_history("/tmp/history.txt")

  

# final test
  print('   final test ...')
  test_batch_size = 10       # size of the batches
  test_accuracy = 0
  test_auc = 0
  nb_iterations = 200
  data.test_iterator = 0
  for _ in range( nb_iterations ) :
      batch_test = data.get_batch_test(batch_size=test_batch_size, crop = False, random_flip_flop = True, random_rotate = True)
      feed_dict = {x:batch_test[0], y_: batch_test[1], keep_prob: 1.0}
      test_accuracy += accuracy.eval(feed_dict)
      test_auc += auc.eval(feed_dict)
            
  test_accuracy /= nb_iterations
  print("   test accuracy %g"%test_accuracy)

  test_auc /= nb_iterations
  print("   test accuracy %g"%test_auc)


#batch_test = data.get_batch_test(max_images=50)
#print("   test accuracy %g"%accuracy.eval(feed_dict={x: batch_test[0], 
#                                                     y_: batch_test[1], 
#                                                     keep_prob: 1.0}))

# done
print("   computation time (cpu) :",time.strftime("%H:%M:%S", time.gmtime(time.clock()-start_clock)))
print("   computation time (real):",time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
print('   done.')


# test some images
# print('   do some test on doctored images.')
# final_prediction = tf.argmax(y_conv,1)
# images = il.get_image_filename_from_dir('/home/nozick/Dropbox/deepLearning/keras/cg_pi/data/test') 

# for im in images :
    
#     random_prefix = ''.join(random.choice('0123456789ABCDEF') for i in range(7))
#     im.save('/tmp/' + random_prefix + '_image.jpg')  
#     print('random prefix =',random_prefix)
       
#     # for each sub image
#     for i in range(0, im.size[0]-image_size, image_size) :
#         for j in range(0, im.size[1]-image_size, image_size) :
#             box = (i, j, i+image_size, j+image_size)
#             sub_im = im.crop(box)
#             sub_im = np.asarray(sub_im)
#             sub_im = sub_im[:,:,1]
#             sub_im = sub_im.astype(np.float32) / 255.
#             sub_im = sub_im.reshape(1,image_size, image_size, 1)
            
#             feed_dict = {x:sub_im, keep_prob: 1.0}
#             result = final_prediction.eval(feed_dict) 
#             #print('result = ', result[0])
#             if (result[0] == 0) :
#                 draw = ImageDraw.Draw(im)
#                 box = (i, j, i+image_size, j+image_size)  
#                 draw.arc(box, 0, 360)
#                 # simulate thinckness = 2
#                 box = (i+2, j+2, i+image_size-2, j+image_size-2)  
#                 draw.arc(box, 0, 360)
#                 del draw
                
#     # save the images
#     im.save('/tmp/' + random_prefix + '_cnn.jpg')    

# print('   done.')












 