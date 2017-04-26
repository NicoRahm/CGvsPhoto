
print("   reset python interpreter ...")
import os
clear = lambda: os.system('clear')
clear()
import time
import random
import image_loader as il
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

# computation time tick
start_clock = time.clock()
start_time = time.time()

# seed initialisation
print("\n   random initialisation ...")
random_seed = int(time.time() % 10000 ) 
random.seed(random_seed)  # for reproducibility
print('   random seed =', random_seed)

folder_ckpt = '/work/smg/v-nicolas/weights/'
dir_summaries = '/home/smg/v-nicolas/summaries/'


# tool functions
print('   python function setup')

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
  return(h)

def gaussian_func(mu, x, n, sigma):
  gauss = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
  return(tf.reduce_sum(gauss.pdf(x)/n))


def gaussian_kernel(x, nbins = 8, values_range = [0, 1], sigma = 0.1,image_size = 100):
  mu_list = np.float32(np.linspace(values_range[0], values_range[1], nbins + 1))
  n = np.float32(image_size**2)
  function_to_map = lambda m : gaussian_func(m, x, n, sigma)
  return(tf.map_fn(function_to_map, mu_list))

def plot_gaussian_kernel(nbins = 8, values_range = [0, 1], sigma = 0.1):

  r = values_range[1] - values_range[0]
  mu_list = []
  for i in range(nbins+1):
    mu_list.append(values_range[0] + i*r/(nbins+1))

  range_plot = np.linspace(values_range[0], values_range[1], 100)

  plt.figure()
  for mu in mu_list:
    plt.plot(range_plot, np.exp(-(range_plot-mu)**2/(sigma**2)))
  plt.title("Gaussian kernels used for estimating the histograms")
  plt.show()

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



def validation_testing(x, y_, keep_prob, accuracy, hist, nbins, it,
                       data, nb_iterations = 20, batch_size = 50,
                       plot_histograms = False, range_hist = [0.,1.], 
                       selected_hist_nb = 4):

  validation_batch_size = batch_size 
  validation_accuracy = 0
  # validation_auc = 0
  data.validation_iterator = 0

  if plot_histograms:
    nb_CGG = 0
    hist_CGG = [np.zeros((nbins+1,)) for i in range(selected_hist_nb)]
    nb_real = 0
    hist_real = [np.zeros((nbins+1,)) for i in range(selected_hist_nb)]

  for _ in range( nb_iterations ) :
    batch_validation = data.get_batch_validation(batch_size=validation_batch_size, 
                                                 crop = False, 
                                                 random_flip_flop = True, 
                                                 random_rotate = True)
    feed_dict = {x: batch_validation[0], 
                 y_: batch_validation[1], 
                 keep_prob: 1.0}
    validation_accuracy += accuracy.eval(feed_dict)

    
    if plot_histograms:
      # Computing the mean histogram for each class
      hist_plot = hist.eval(feed_dict)
      for k in range(validation_batch_size): 
        if batch_validation[1][k][0] == 0.:
          nb_real +=1
          is_real = True
        else:
          nb_CGG += 1
          is_real = False
        for j in range(selected_hist_nb):
          for l in range(nbins+1): 
            if is_real:
              hist_real[j][l] += hist_plot[k,j,l]
            else:
              hist_CGG[j][l] += hist_plot[k,j,l]
              
      for p in range(selected_hist_nb):
        hist_CGG[p] /= nb_CGG
        hist_real[p] /= nb_real

  if plot_histograms:
    # Plotting mean histogram for CGG
    fig = plt.figure(1)
    for k in range(selected_hist_nb):
      plt.subplot(selected_hist_nb/2, 2, k+1)
      plt.bar(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_CGG[k], width = 1/(nbins + 1))
      plt.plot(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_CGG[k], 'r')
      fig.suptitle("Mean histogram for CGG", fontsize=14)
    plt.show()

    # Plotting mean histogram for Real
    fig = plt.figure(2)
    for k in range(selected_hist_nb):
      plt.subplot(selected_hist_nb/2, 2, k+1)
      plt.bar(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_real[k], width = 1/(nbins + 1))
      plt.plot(np.linspace(range_hist[0], range_hist[1],nbins+1), hist_real[k], 'r')
      fig.suptitle("Mean histogram for Real", fontsize=14)
    plt.show()

  validation_accuracy /= nb_iterations
  print("     step %d, training accuracy %g (%d validations tests)"%(it, validation_accuracy, validation_batch_size*nb_iterations))

  return(validation_accuracy)




def train_classifier(database_path, image_size, nb_train_batch,
                     nb_test_batch, nb_validation_batch, 
                     batch_size = 50, clf = None, all_summaries = False):

  # start process
  print('   tensorFlow version: ', tf.__version__)

  # load data
  print('   import data : image_size = ' + 
        str(image_size) + 'x' + str(image_size) + '...')

  data = il.Database_loader(database_path, image_size, 
                            proportion = 1, only_green=True)

  
  run_name = input("   Choose a name for the run : ")

  path_save = folder_ckpt + run_name


  print('   create model ...')
  # input layer. One entry is a float size x size, 3-channels image. 
  # None means that the number of such vector can be of any lenght.

  graph = tf.Graph()

  with graph.as_default():

    with tf.name_scope('Input_Data'):
      x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])

      # reshape the input data:
      x_image = tf.reshape(x, [-1,image_size, image_size, 1])
      

    # first conv net layer
    nb_conv1 = 32
    filter_size1 = 3

    with tf.name_scope('Conv1'):

      with tf.name_scope('Weights'):
        W_conv1 = weight_variable([filter_size1, filter_size1, 1, nb_conv1], seed = random_seed)
      with tf.name_scope('Bias'):
        b_conv1 = bias_variable([nb_conv1])


      # relu on the conv layer
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, 
                           name = 'Activated_1')

    # second conv 
    nb_conv2 = 64
    filter_size2 = 3
    with tf.name_scope('Conv2'):
      with tf.name_scope('Weights'):
        W_conv2 = weight_variable([filter_size2, filter_size2, nb_conv1, nb_conv2])
      with tf.name_scope('Bias'):
        b_conv2 = bias_variable([nb_conv2])

      h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2, 
                           name = 'Activated_2')

    # Histograms
    nbins = 10
    nb_filters = nb_conv2
    size_hist = (nbins + 1)*nb_filters
    range_hist = [0,1]
    sigma = 0.07

    # plot_gaussian_kernel(nbins = nbins, values_range = range_hist, sigma = 0.1)

    with tf.name_scope('Gaussian_Histogram'): 
      hist = classic_histogram_gaussian(h_conv2, k = nb_filters, nbins = nbins, values_range = range_hist, sigma = sigma)
      tf.summary.tensor_summary('hist', hist)

    h_pool_flat = tf.reshape(hist, [-1, size_hist], name = "Flatten_Hist")


    # Densely Connected Layer
    # we add a fully-connected layer with 1024 neurons 
    with tf.variable_scope('Dense1'):
      with tf.name_scope('Weights'):
        W_fc1 = weight_variable([size_hist, 1024])
      with tf.name_scope('Bias'):
        b_fc1 = bias_variable([1024])
      # put a relu
      h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1, name = 'activated')

    # dropout
    with tf.name_scope('Dropout1'):
      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    with tf.variable_scope('Readout'):
      with tf.name_scope('Weights'):
        W_fc3 = weight_variable([1024, data.nb_class])
      with tf.name_scope('Bias'):
        b_fc3 = bias_variable([data.nb_class])
      y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3


    # support for the learning label
    y_ = tf.placeholder(tf.float32, [None, data.nb_class])




    # Define loss (cost) function and optimizer
    print('   setup loss function and optimizer ...')

    # softmax to have normalized class probabilities + cross-entropy
    with tf.name_scope('cross_entropy'):

      softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
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
      tf.summary.scalar('accuracy', accuracy)

    # with tf.name_scope('AUC'):
    #   auc = tf.metrics.auc(tf.argmax(y_,1), tf.argmax(y_conv,1))
    #   tf.summary.scalar('AUC', auc)

    if all_summaries:
      with tf.name_scope('Image_Visualization'):
        tf.summary.image('Input_Data', x_image)
      with tf.name_scope('Weights'):
        variable_summaries(W_conv1)
        variable_summaries(W_conv2)
        variable_summaries(W_fc1)
        variable_summaries(W_fc3)
      with tf.name_scope('Bias'):
        variable_summaries(b_conv1)
        variable_summaries(b_conv2)
        variable_summaries(b_fc1)  
        variable_summaries(b_fc3)  
      with tf.variable_scope('Conv_visualization'):
        tf.summary.image('conv1/filters', W_conv1[:,:,:,0:1])
        tf.summary.image('conv2/filters', W_conv2[:,:,:,0:1])
      tf.summary.histogram('Activated_Conv_1', h_conv1)
      tf.summary.histogram('Activated_Conv_2', h_conv2)
      tf.summary.histogram('Activated_Fully_Connected', h_fc1)


  # start a session
  print('   start session ...')
  with tf.Session(graph=graph) as sess:

    merged = tf.summary.merge_all()

    if not os.path.exists(dir_summaries + run_name):
      os.mkdir(dir_summaries + run_name)


    train_writer = tf.summary.FileWriter(dir_summaries + run_name,
                                         sess.graph)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    saver = tf.train.Saver()
    print('   variable initialization ...')

    restore_weigths = input("Restore weight from previous session ? (y/N) : ")

    if restore_weigths == 'y':
      file_to_restore = input("Name of the file to restore (Directory : " + 
                              folder_ckpt + ') : ')
      saver.restore(sess, folder_ckpt + file_to_restore)
      print('   Model restored')
      

    # Train
    print('   train ...')
    for i in range(nb_train_batch):
      
        # evry 100 batches, test the accuracy
        if i%10 == 0 :
            
            validation_testing(x, y_, keep_prob, accuracy, hist, nbins, i,
                               data, nb_iterations = nb_validation_batch, 
                               batch_size = batch_size, 
                               plot_histograms = False)
            
        # regular training
        batch = data.get_next_train_batch(batch_size, False, True, True)
        feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.85}
        summary, _ = sess.run([merged, train_step], feed_dict = feed_dict)
        train_writer.add_summary(summary, i)

        # Saving weights every 100 batches
        if i%100 == 0:
          path_save_batch = path_save + str(i) + ".ckpt"
          print('   saving weights in file : ' + path_save_batch)
          saver.save(sess, path_save_batch)
          print('   OK')

  # final test
    print('   final test ...')
    test_accuracy = 0
    # test_auc = 0
    nb_iterations = 20
    data.test_iterator = 0
    for _ in range( nb_iterations ) :
        batch_test = data.get_batch_test(batch_size, False, True, True)
        feed_dict = {x:batch_test[0], y_: batch_test[1], keep_prob: 1.0}
        test_accuracy += accuracy.eval(feed_dict)
        # test_auc += sess.run(auc, feed_dict)[0]

              
    test_accuracy /= nb_iterations
    print("   test accuracy %g"%test_accuracy)

    # test_auc /= (nb_iterations - 1)
    # print("   test AUC %g"%test_auc)

  # done
  print("   computation time (cpu) :",time.strftime("%H:%M:%S", time.gmtime(time.clock()-start_clock)))
  print("   computation time (real):",time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
  print('   done.')

  return(graph)

def test_all_images(graph): 

  print('   start session ...')
  with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver()
    print('   variable initialization ...')
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    file_to_restore = input("Name of the file to restore (Directory : " + 
                            folder_ckpt + ') : ')
    saver.restore(sess, folder_ckpt + file_to_restore)

if __name__ == '__main__':

  database_path = '/work/smg/v-nicolas/Test_DB_200'
  image_size = 200
  nb_train_batch = 2000
  nb_test_batch = 40
  nb_validation_batch = 20

  clf = train_classifier(database_path, image_size, nb_train_batch,
                         nb_test_batch, nb_validation_batch, 
                         batch_size = 50, clf = None, 
                         all_summaries = False)

  test_all_images(clf)