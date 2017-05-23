
print("   reset python interpreter ...")
import os
clear = lambda: os.system('clear')
clear()
import time
import random
import image_loader as il
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import csv

import numpy as np

GPU = '/gpu:0'

config = ''
config = 'server'

if config != 'server':
  from sklearn.metrics import roc_curve
  from sklearn.metrics import auc
  from sklearn.metrics import accuracy_score as acc
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  from sklearn.svm import SVC
  # from sklearn.ensemble import ExtraTreesClassifier

# computation time tick
start_clock = time.clock()
start_time = time.time()

# seed initialisation
print("\n   random initialisation ...")
random_seed = int(time.time() % 10000 ) 
random.seed(random_seed)  # for reproducibility
print('   random seed =', random_seed)

if config == 'server':
  folder_ckpt = '/work/smg/v-nicolas/weights/'
  dir_summaries = '/home/smg/v-nicolas/summaries/'
  visualization_dir = '/home/smg/v-nicolas/visualization/'
else:
  folder_ckpt = '/home/nicolas/Documents/weights/'
  dir_summaries = '/home/nicolas/Documents/summaries/'
  visualization_dir = '/home/nicolas/Documents/visualization/'


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
  initial = tf.truncated_normal(shape, mean = 0.5, stddev=0.1, seed = random_seed)
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

def gaussian_func(mu, x, n, sigma, xmax = 1):
  xmax = np.float32(xmax)
  gauss = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
  # return(tf.reduce_sum(gauss.pdf(xmax - tf.nn.relu(xmax - x))/n))
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

  range_plot = np.linspace(values_range[0]-0.1, values_range[1]+0.1, 1000)

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

def stat(x):
  return(tf.stack([tf.reduce_mean(x), tf.reduce_min(x), tf.reduce_max(x), tf.reduce_mean((x - tf.reduce_mean(x))**2)]))

def compute_stat(x, k):
  function_to_map = lambda y: tf.stack([stat(y[:,:,i]) for i in range(k)])
  res = tf.map_fn(function_to_map, x)
  return(res)

class Model:

  def __init__(self, database_path, image_size, nbins = 10, 
               batch_size = 50, histograms = True, stats = False, 
               all_summaries = False, using_GPU = False):

    print('   tensorFlow version: ', tf.__version__)

    self.database_path = database_path
    self.image_size = image_size
    self.batch_size = batch_size
    self.all_summaries = all_summaries
    self.nbins = nbins
    self.histograms = histograms
    self.stats = stats
    self.using_GPU = using_GPU

    self.import_database()
    if using_GPU:
      with tf.device(GPU):
        self.create_graph(nb_class = self.nb_class, 
                          histograms = self.histograms,
                          stats = self.stats,
                          all_summaries = self.all_summaries)
    else: 
      self.create_graph(nb_class = self.nb_class, 
                          histograms = self.histograms,
                          stats = self.stats,
                          all_summaries = self.all_summaries)



  def import_database(self): 
    # load data
    print('   import data : image_size = ' + 
        str(self.image_size) + 'x' + str(self.image_size) + '...')
    self.data = il.Database_loader(database_path, image_size, 
                                   proportion = 1, only_green=True)
    self.nb_class = self.data.nb_class

  def create_graph(self, nb_class, histograms = True, stats = False,
                   all_summaries = False): 

    print('   create model ...')
    # input layer. One entry is a float size x size, 3-channels image. 
    # None means that the number of such vector can be of any lenght.

    if (histograms): 
      print('    Model with histograms.')

    else: 
      print('    Model without histograms.')
    graph = tf.Graph()

    with graph.as_default():

      with tf.name_scope('Input_Data'):
        x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
        self.x = x
        # reshape the input data:
        x_image = tf.reshape(x, [-1,image_size, image_size, 1])
        with tf.name_scope('Image_Visualization'):
          tf.summary.image('Input_Data', x_image)
        

      # first conv net layer
      nb_conv1 = 32
      self.nb_conv1 = nb_conv1
      filter_size1 = 3

      with tf.name_scope('Conv1'):

        with tf.name_scope('Weights'):
          W_conv1 = weight_variable([filter_size1, filter_size1, 1, nb_conv1], seed = random_seed)
          self.W_conv1 = W_conv1
        with tf.name_scope('Bias'):
          b_conv1 = bias_variable([nb_conv1])


        # relu on the conv layer
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, 
                             name = 'Activated_1')

      # second conv 
      nb_conv2 = 64
      self.nb_conv2 = nb_conv2
      filter_size2 = 3
      with tf.name_scope('Conv2'):
        with tf.name_scope('Weights'):
          W_conv2 = weight_variable([filter_size2, filter_size2, nb_conv1, nb_conv2])
          self.W_conv2 = W_conv2
        with tf.name_scope('Bias'):
          b_conv2 = bias_variable([nb_conv2])

        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2, 
                             name = 'Activated_2')

        self.h_conv2 = h_conv2

        tf.summary.image('Filtered_image_1', h_conv1[:,:,:,0:1])
        tf.summary.image('Filtered_image_2', h_conv1[:,:,:,1:2])
        tf.summary.image('Filtered_image_3', h_conv1[:,:,:,2:3])

      # m_pool = max_pool_2x2(h_conv2)

      # nb_conv3 = 64

      # filter_size3 = 3
      # with tf.name_scope('Conv3'):
      #   with tf.name_scope('Weights'):
      #     W_conv3 = weight_variable([filter_size3, filter_size3, nb_conv2, nb_conv3])
      #   with tf.name_scope('Bias'):
      #     b_conv3 = bias_variable([nb_conv3])

      #   h_conv3 = tf.nn.relu(conv2d(m_pool, W_conv3) + b_conv3, 
      #                        name = 'Activated_3')

      



      nb_filters = nb_conv2
      if histograms:
        # Histograms
        nbins = self.nbins
        size_flat = (nbins + 1)*nb_filters

        range_hist = [0,1]
        sigma = 0.05

        # plot_gaussian_kernel(nbins = nbins, values_range = range_hist, sigma = sigma)

        with tf.name_scope('Gaussian_Histogram'): 
          hist = classic_histogram_gaussian(h_conv2, k = nb_filters, 
                                            nbins = nbins, 
                                            values_range = range_hist, 
                                            sigma = sigma)
          self.hist = hist
          # tf.summary.tensor_summary('hist', hist)

        flatten = tf.reshape(hist, [-1, size_flat], name = "Flatten_Hist")
        self.flatten = flatten

      else: 

        if stats: 

          s = compute_stat(h_conv2, nb_filters)
          size_flat = nb_filters*4
          flatten = tf.reshape(s, [-1, size_flat], name = "Flattend_Stat")
          self.hist = s
        else:
          m_pool = max_pool_2x2(h_conv2)
          size_flat = int(nb_filters*(image_size**2)/4)
          flatten = tf.reshape(m_pool, [-1, size_flat], name = "Flatten_MPool")

        self.flatten = flatten
      # Densely Connected Layer
      # we add a fully-connected layer with 1024 neurons 
      with tf.variable_scope('Dense1'):
        with tf.name_scope('Weights'):
          W_fc1 = weight_variable([size_flat, 1024])
        with tf.name_scope('Bias'):
          b_fc1 = bias_variable([1024])
        # put a relu
        h_fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1, 
                           name = 'activated')

      # dropout
      with tf.name_scope('Dropout1'):
        keep_prob = tf.placeholder(tf.float32)
        self.keep_prob = keep_prob
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      self.h_fc1 = h_fc1

      # readout layer
      with tf.variable_scope('Readout'):
        with tf.name_scope('Weights'):
          W_fc3 = weight_variable([1024, nb_class])
        with tf.name_scope('Bias'):
          b_fc3 = bias_variable([nb_class])
        y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

      self.y_conv = y_conv

      # support for the learning label
      y_ = tf.placeholder(tf.float32, [None, nb_class])
      self.y_ = y_



      # Define loss (cost) function and optimizer
      print('   setup loss function and optimizer ...')

      # softmax to have normalized class probabilities + cross-entropy
      with tf.name_scope('cross_entropy'):

        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
        with tf.name_scope('total'):
          cross_entropy_mean = tf.reduce_mean(softmax_cross_entropy)

      tf.summary.scalar('cross_entropy', cross_entropy_mean)

      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)

      self.train_step = train_step
      print('   test ...')
      # 'correct_prediction' is a function. argmax(y, 1), here 1 is for the axis number 1
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

      # 'accuracy' is a function: cast the boolean prediction to float and average them
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

      self.accuracy = accuracy

      # with tf.name_scope('AUC'):
      #   auc = tf.metrics.auc(tf.argmax(y_,1), tf.argmax(y_conv,1))
      #   tf.summary.scalar('AUC', auc)

      if all_summaries:
        # with tf.name_scope('Image_Visualization'):
          # tf.summary.image('Input_Data', x_image)
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

    self.graph = graph

  def validation_testing(self, it, nb_iterations = 20, batch_size = 50,
                         plot_histograms = False, range_hist = [0.,1.], 
                         selected_hist_nb = 8, run_name = '',
                         save_filters = True):

    if save_filters: 
      
      nb_height = 8
      nb_width = int(self.nb_conv1/nb_height)

      img, axes = plt.subplots(nrows = nb_width, ncols = nb_height)
      gs1 = gridspec.GridSpec(nb_height, nb_width)
      for i in range(self.nb_conv1):
        ax1 = plt.subplot(gs1[i])
        ax1.axis('off')
        im = plt.imshow(self.W_conv1[:,:,0,i].eval())
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])     

      img.subplots_adjust(right=0.8)
      cbar_ax = img.add_axes([0.85, 0.15, 0.05, 0.7])
      cbar = img.colorbar(im, ticks=[-0.5, 0, 0.5], cax=cbar_ax)
      cbar.ax.set_yticklabels(['< -0.5', '0', '> 0.5'])
      plt.show(img)
      plt.close()     

    validation_batch_size = batch_size 
    validation_accuracy = 0
    # validation_auc = 0
    self.data.validation_iterator = 0

    if plot_histograms:
      nb_CGG = 0
      hist_CGG = [np.zeros((self.nbins+1,)) for i in range(selected_hist_nb)]
      nb_real = 0
      hist_real = [np.zeros((self.nbins+1,)) for i in range(selected_hist_nb)]

    for _ in range( nb_iterations ) :
      batch_validation = self.data.get_batch_validation(batch_size=validation_batch_size, 
                                                        crop = False, 
                                                        random_flip_flop = True, 
                                                        random_rotate = True)
      feed_dict = {self.x: batch_validation[0], 
                   self.y_: batch_validation[1], 
                   self.keep_prob: 1.0}
      validation_accuracy += self.accuracy.eval(feed_dict)

      
      if plot_histograms:
        # Computing the mean histogram for each class
        hist_plot = self.hist.eval(feed_dict)
        for k in range(validation_batch_size): 
          if batch_validation[1][k][0] == 1.:
            nb_real +=1
            is_real = True
          else:
            nb_CGG += 1
            is_real = False
          for j in range(selected_hist_nb):
            for l in range(self.nbins+1): 
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
        plt.bar(np.linspace(range_hist[0], range_hist[1], self.nbins+1), 
                            hist_CGG[k], width = 1/(self.nbins + 1))
        plt.plot(np.linspace(range_hist[0], range_hist[1], self.nbins+1), 
                             hist_CGG[k], 'r')
        fig.suptitle("Mean histogram for CGG", fontsize=14)
      plt.savefig('/home/smg/v-nicolas/visualization/histograms/' + run_name + '_cgg_'+ str(it) + '.png')
      plt.close()

      # Plotting mean histogram for Real
      fig = plt.figure(2)
      for k in range(selected_hist_nb):
        plt.subplot(selected_hist_nb/2, 2, k+1)
        plt.bar(np.linspace(range_hist[0], range_hist[1], self.nbins+1), 
                            hist_real[k], width = 1/(self.nbins + 1))
        plt.plot(np.linspace(range_hist[0], range_hist[1],self.nbins+1), 
                             hist_real[k], 'r')
        fig.suptitle("Mean histogram for Real", fontsize=14)
      plt.savefig('/home/smg/v-nicolas/visualization/histograms/' + run_name + '_real_'+ str(it) + '.png')
      plt.close()



    validation_accuracy /= nb_iterations
    print("     step %d, training accuracy %g (%d validations tests)"%(it, validation_accuracy, validation_batch_size*nb_iterations))
    return(validation_accuracy)


  def train(self, nb_train_batch, nb_test_batch, 
            nb_validation_batch, batch_size = 50,
            save_filters = False):
    
    run_name = input("   Choose a name for the run : ")
    path_save = folder_ckpt + run_name
    acc_name = dir_summaries + "validation_accuracy.csv"


    # computation time tick
    start_clock = time.clock()
    start_time = time.time()
    batch_clock = None

    # start a session
    print('   start session ...')
    with tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=self.using_GPU)) as sess:

      merged = tf.summary.merge_all()
      
      if not os.path.exists(dir_summaries + run_name):
        os.mkdir(dir_summaries + run_name)


      train_writer = tf.summary.FileWriter(dir_summaries + run_name,
                                           sess.graph)

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      saver = tf.train.Saver()
      print('   variable initialization ...')

      restore_weigths = input("\nRestore weight from previous session ? (y/N) : ")

      if restore_weigths == 'y':
        file_to_restore = input("\nName of the file to restore (Directory : " + 
                                folder_ckpt + ') : ')
        saver.restore(sess, folder_ckpt + file_to_restore)
        print('\n   Model restored\n')
        

      # Train
      print('   train ...')
      validation_accuracy = []
      for i in range(nb_train_batch):
        
          # evry 100 batches, test the accuracy
          if i%10 == 0 :
              
              if i%100 == 0:
                plot_histograms = False
              else:
                plot_histograms = False

              v = self.validation_testing(i, nb_iterations = nb_validation_batch, 
                                      batch_size = batch_size, 
                                      plot_histograms = plot_histograms,
                                      run_name = run_name,
                                      save_filters = save_filters)
              validation_accuracy.append(v)
              
          # regular training
          batch = self.data.get_next_train_batch(batch_size, False, True, True)
          feed_dict = {self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.65}
          summary, _ = sess.run([merged, self.train_step], feed_dict = feed_dict)
          train_writer.add_summary(summary, i)

          # Saving weights every 100 batches
          if i%100 == 0:

            path_save_batch = path_save + str(i) + ".ckpt"
            print('   saving weights in file : ' + path_save_batch)
            saver.save(sess, path_save_batch)
            print('   OK')
            if batch_clock is not None: 
              time_elapsed = (time.time()-batch_clock)
              print('   Time last 100 batchs : ', time.strftime("%H:%M:%S",time.gmtime(time_elapsed)))
              remaining_time = time_elapsed * int((nb_train_batch - i)/100)
              print('   Remaining time : ', time.strftime("%H:%M:%S",time.gmtime(remaining_time)))
            batch_clock = time.time()
      print('   saving validation accuracy...')
      file = open(acc_name, 'w', newline='')

      try:
          writer = csv.writer(file)
       
          for v in validation_accuracy:
            writer.writerow([str(v)])
      finally:

          file.close()
          print('   done.')

      plt.figure()
      plt.plot(np.linspace(0,nb_train_batch,10), validation_accuracy)
      plt.title("Validation accuracy during training")
      plt.xlabel("Training batch")
      plt.ylabel("Validation accuracy")
    # final test
      print('   final test ...')
      test_accuracy = 0
      # test_auc = 0
      nb_iterations = 20
      self.data.test_iterator = 0
      for _ in range( nb_iterations ) :
          batch_test = self.data.get_batch_test(batch_size, False, True, True)
          feed_dict = {self.x:batch_test[0], self.y_: batch_test[1], self.keep_prob: 1.0}
          test_accuracy += self.accuracy.eval(feed_dict)
          # test_auc += sess.run(auc, feed_dict)[0]

                
      test_accuracy /= nb_iterations
      print("   test accuracy %g"%test_accuracy)

      # test_auc /= (nb_iterations - 1)
      # print("   test AUC %g"%test_auc)

    # done
    print("   computation time (cpu) :",time.strftime("%H:%M:%S", time.gmtime(time.clock()-start_clock)))
    print("   computation time (real):",time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
    print('   done.')


  def show_histogram(self):

    with tf.Session(graph=self.graph) as sess:

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      saver = tf.train.Saver()
      print('   variable initialization ...')

      file_to_restore = input("\nName of the file to restore (Directory : " + 
                                folder_ckpt + ') : ')
      saver.restore(sess, folder_ckpt + file_to_restore)
      print('\n   Model restored\n')

      batch = self.data.get_next_train_batch(self.batch_size, False, True, True)
      feed_dict = {self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0}
      conv = self.h_conv2.eval(feed_dict = feed_dict)

      for i in range(self.batch_size):
        plt.figure()

        plt.hist(np.reshape(conv[i,:,:,0], (self.image_size*self.image_size,)))
        plt.show()

  def lda_training(self, nb_train_batch, nb_test_batch):

    self.lda_classifier = LinearDiscriminantAnalysis()

    # start a session
    print('   start session ...')
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      print('   variable initialization ...')
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = input("\nName of the file to restore (Directory : " + 
                              folder_ckpt + ') : ')
      saver.restore(sess, folder_ckpt + file_to_restore)

      # training the LDA classifier
      features = []
      labels = []
      for i in range(nb_train_batch):
        if (i%10 == 0):
          print("Computing features for training batch " + str(i) + '/' + str(nb_train_batch))

        batch = self.data.get_next_train_batch(self.batch_size, False, True, True)
        feed_dict = {self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0}
        h = self.flatten.eval(feed_dict = feed_dict)
        features.append(h)
        labels.append(np.argmax(np.array(batch[1]), 1))
      
      features = np.reshape(np.array(features), (self.batch_size*nb_train_batch, features[0].shape[1])) 
      labels = np.reshape(np.array(labels), (self.batch_size*nb_train_batch,)) 
      print(features.shape)
      print(labels.shape)
      self.lda_classifier.fit(features, labels)


      print('   Testing ...')
      # test_auc = 0
      features_test = []
      labels_test = []
      for _ in range(nb_test_batch) :
        batch_test = self.data.get_batch_test(self.batch_size, False, True, True)
        feed_dict = {self.x:batch_test[0], self.y_: batch_test[1], self.keep_prob: 1.0}
        h = self.flatten.eval(feed_dict = feed_dict)
        features_test.append(h)
        labels_test.append(np.argmax(np.array(batch_test[1]), 1))

      features_test = np.reshape(np.array(features_test), (self.batch_size*nb_test_batch, features_test[0].shape[1])) 
      labels_test = np.reshape(np.array(labels_test), (self.batch_size*nb_test_batch,)) 

      labels_pred = self.lda_classifier.predict(features_test)

      test_accuracy = acc(labels_pred, labels_test)
      print("   test accuracy %g"%test_accuracy)

  def svm_training(self, nb_train_batch, nb_test_batch):

    # self.svm_classifier = ExtraTreesClassifier(n_estimators = 200, max_depth = 6)
    self.svm_classifier = SVC()
    # start a session
    print('   start session ...')
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      print('   variable initialization ...')
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = input("\nName of the file to restore (Directory : " + 
                              folder_ckpt + ') : ')
      saver.restore(sess, folder_ckpt + file_to_restore)

      # training the LDA classifier
      features = []
      labels = []
      for i in range(nb_train_batch):
        if (i%10 == 0):
          print("Computing features for training batch " + str(i) + '/' + str(nb_train_batch))

        batch = self.data.get_next_train_batch(self.batch_size, False, True, True)
        feed_dict = {self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0}
        h = self.flatten.eval(feed_dict = feed_dict)
        features.append(h)
        labels.append(np.argmax(np.array(batch[1]), 1))
      
      features = np.reshape(np.array(features), (self.batch_size*nb_train_batch, features[0].shape[1])) 
      labels = np.reshape(np.array(labels), (self.batch_size*nb_train_batch,)) 
      print(features.shape)
      print(labels.shape)
      self.svm_classifier.fit(features, labels)


      print('   Testing ...')
      # test_auc = 0
      features_test = []
      labels_test = []
      for _ in range(nb_test_batch) :
        batch_test = self.data.get_batch_test(self.batch_size, False, True, True)
        feed_dict = {self.x:batch_test[0], self.y_: batch_test[1], self.keep_prob: 1.0}
        h = self.flatten.eval(feed_dict = feed_dict)
        features_test.append(h)
        labels_test.append(np.argmax(np.array(batch_test[1]), 1))

      features_test = np.reshape(np.array(features_test), (self.batch_size*nb_test_batch, features_test[0].shape[1])) 
      labels_test = np.reshape(np.array(labels_test), (self.batch_size*nb_test_batch,)) 

      labels_pred = self.svm_classifier.predict(features_test)

      test_accuracy = acc(labels_pred, labels_test)
      print("   test accuracy %g"%test_accuracy)


  def test_total_images(self, test_data_path, nb_images, 
                        minibatch_size = 25, decision_rule = 'majority_vote',
                        show_images = False,
                        save_images = False,
                        only_green = True): 

    valid_decision_rule = ['majority_vote', 'weighted_vote']
    if decision_rule not in valid_decision_rule:
      raise NameError(decision_rule + ' is not a valid decision rule.')
    if(save_images):
      test_name = input("   Choose a name for the test : ")
      if not os.path.exists(visualization_dir + test_name):
        os.mkdir(visualization_dir + test_name)

    if not only_green:
      print('   No visualization when testing all channels...')
      show_images = False
      save_images = False
    print('   start session ...')
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      print('   variable initialization ...')
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = input("\nName of the file to restore (Directory : " + 
                              folder_ckpt + ') : ')
      saver.restore(sess, folder_ckpt + file_to_restore)

      data_test = il.Test_loader(test_data_path, subimage_size = self.image_size, only_green = only_green)

      y = []
      scores = []
      tp = 0
      fp = 0
      nb_CGG = 0
      accuracy = 0
      for i in range(nb_images):
        batch, label, width, height, original = data_test.get_next_image()
        if not only_green: 
          batch = np.reshape(batch, (batch.shape[0]*3, batch.shape[1], batch.shape[2],1))
        batch_size = batch.shape[0]
        j = 0
        prediction = 0
        labels = []
        diff = []
        
        while j < batch_size:

          feed_dict = {self.x: batch[j:j+minibatch_size], self.keep_prob: 1.0}
          pred = self.y_conv.eval(feed_dict)
          label_image = np.argmax(pred, 1)
          d = np.max(pred, 1) - np.min(pred, 1)

          for k in range(d.shape[0]):
            diff.append(np.round(d[k], 1))

          if decision_rule == 'majority_vote':
            prediction += np.sum(label_image)
          if decision_rule == 'weighted_vote':
            prediction += np.sum(2*d*(label_image - 0.5))

          for l in label_image:
            labels.append(data_test.image_class[l])
          j+=minibatch_size

         
        if config != 'server':
          if(label == 'Real'):
            y.append(-1)
          else:
            y.append(1)

          scores.append(prediction)

        diff = np.array(diff)
        if decision_rule == 'majority_vote':
          prediction = data_test.image_class[int(np.round(prediction/batch_size))]
        if decision_rule == 'weighted_vote':
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
          self.image_visualization(path_save = visualization_dir + test_name, 
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

    if config != 'server':
      fpr, tpr, thresholds = roc_curve(np.array(y), np.array(scores))

      plt.figure()
      lw = 2
      plt.plot(fpr, tpr, color='darkorange')
      plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic curve')
      plt.show()

    print('\n_______________________________________________________')
    print('Final Accuracy : ' + str(round(100*accuracy/(nb_images), 3)) + '%')
    print('Final Precision : ' + str(round(100*tp/(tp + fp), 3)) + '%')
    print('Final Recall : ' + str(round(100*tp/nb_CGG, 3)) + '%')
    print('Final AUC : ' + str(round(100*auc(fpr, tpr), 3)) + '%')
    print('_______________________________________________________\n')

  def image_visualization(self, path_save, file_name, images, labels_pred, 
                          true_label, width, height, diff, original = None,
                          show_images = False, save_images = False,
                          prob_map = False, save_original = False):

    nb_width = int(width/self.image_size)
    nb_height = int(height/self.image_size)

    img = plt.figure(figsize = (nb_width, nb_height))
    
    gs1 = gridspec.GridSpec(nb_height, nb_width)
    
    
    for i in range(len(images)):

      cdict_green = {'red': ((0.0,0.0,0.0),
                             (1.0,1.0 - diff[i]/4,1.0 - diff[i]/4)),
                     'blue': ((0.0,0.0,0.0),
                              (1.0,1.0 - diff[i]/4,1.0 - diff[i]/4)),
                     'green': ((0.0,0.0,0.0),
                               (1.0,1.0,1.0))}

      cdict_red = {'red': ((0.0,0.0,0.0),
                               (1.0,1.0,1.0)),
                    'blue': ((0.0,0.0,0.0),
                                (1.0,1.0 - diff[i]/4,1.0 - diff[i]/4)),
                    'green': ((0.0,0.0,0.0),
                                 (1.0,1.0 - diff[i]/4,1.0 - diff[i]/4))}
      

      ax1 = plt.subplot(gs1[i])
      ax1.axis('off')
      if labels_pred[i] == true_label:
        if diff[i] > 0.4:
          cmap = mcolors.LinearSegmentedColormap('my_green', cdict_green, 100)
        else:
          cmap = 'gray'
      else: 
        if diff[i] > 0.4:
          cmap = mcolors.LinearSegmentedColormap('my_red', cdict_red, 100)
        else:
          cmap = 'gray'
      plt.imshow(images[i,:,:,0], cmap = cmap)
      ax1.set_xticklabels([])
      ax1.set_yticklabels([])
      # ax1.text(40, 50, str(diff[i]))

    gs1.update(wspace=.0, hspace=.0)
    if show_images:
      plt.show(img)
    if save_images:
      plt.savefig(path_save + '/vis_' + file_name + '.png', 
                  bbox_inches='tight',
                  pad_inches=0.0)

    plt.close()

    if save_images:
      if save_original: 
        plt.figure()
        plt.axis('off')
        plt.imshow(original, cmap = 'gray')
        plt.savefig(path_save + '/vis_' + file_name + '_original' + '.png', 
                  bbox_inches='tight',
                  pad_inches=0.0)
      if prob_map: 
        img = plt.figure(figsize = (nb_width, nb_height))

        
      
        gs1 = gridspec.GridSpec(nb_height, nb_width)
        for i in range(len(images)):
          map_im = np.ones((self.image_size, self.image_size))
          map_im[0,0] = 0
          cdict_green = {'red': ((0.0,0.0,0.0),
                             (1.0,1.0 - diff[i]/4,1.0 - diff[i]/4)),
                     'blue': ((0.0,0.0,0.0),
                              (1.0,1.0 - diff[i]/4,1.0 - diff[i]/4)),
                     'green': ((0.0,0.0,0.0),
                               (1.0,1.0,1.0))}

          cdict_red = {'red': ((0.0,0.0,0.0),
                                   (1.0,1.0,1.0)),
                        'blue': ((0.0,0.0,0.0),
                                    (1.0,1.0 - diff[i]/4,1.0 - diff[i]/4)),
                        'green': ((0.0,0.0,0.0),
                                     (1.0,1.0 - diff[i]/4,1.0 - diff[i]/4))}
          
          ax1 = plt.subplot(gs1[i])
          ax1.axis('off')
          if labels_pred[i] == true_label:
            if diff[i] > 0.4:
              cmap = mcolors.LinearSegmentedColormap('my_green', cdict_green, 100)
            else:
              cmap = 'gray'
              map_im = map_im*0.5
          else: 
            if diff[i] > 0.4:
              cmap = mcolors.LinearSegmentedColormap('my_red', cdict_red, 100)
            else:
              cmap = 'gray'
              map_im = map_im*0.5

          plt.imshow(map_im, cmap = cmap)
          ax1.set_xticklabels([])
          ax1.set_yticklabels([])

        gs1.update(wspace=.0, hspace=.0)
        if show_images:
          plt.show(img)
        if save_images:
          plt.savefig(path_save + '/vis_' + file_name + '_probmap' + '.png', 
                      bbox_inches='tight',
                      pad_inches=0.0)

        plt.close()


  def test_splicing(self, data_path, nb_images, save_images, show_images,
                    minibatch_size = 25):

    if(save_images):
      test_name = input("   Choose a name for the test : ")
      path_save = visualization_dir + test_name
      if not os.path.exists(visualization_dir + test_name):
        os.mkdir(visualization_dir + test_name)

    else: 
      path_save = ''

    print('   start session ...')
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      print('   variable initialization ...')
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = input("\nName of the file to restore (Directory : " + 
                              folder_ckpt + ') : ')
      saver.restore(sess, folder_ckpt + file_to_restore)

      data_test = il.Test_loader(data_path, 
                                 subimage_size = self.image_size)

      for i in range(nb_images):
        batch, label, width, height, original = data_test.get_next_image()
        batch_size = batch.shape[0]
        j = 0
        labels = []
        diff = []
        while j < batch_size:
          feed_dict = {self.x: batch[j:j+minibatch_size], self.keep_prob: 1.0}
          pred = self.y_conv.eval(feed_dict)
          label_image = np.argmax(pred, 1)
          d = np.max(pred, 1) - np.min(pred, 1)
          for k in range(d.shape[0]):
            diff.append(np.round(d[k], 1))
          for l in label_image:
            labels.append(data_test.image_class[l])
          j+=minibatch_size

        diff = np.array(diff)

        
        self.image_visualization(path_save = path_save, 
                                 file_name = str(i), 
                                 images = batch, labels_pred = labels, 
                                 true_label = label, width = width, 
                                 height = height, diff = diff,
                                 original = original,
                                 show_images = show_images,
                                 save_images = save_images,
                                 prob_map = save_images, 
                                 save_original= save_images)    


if __name__ == '__main__':

  using_GPU = False

  if config == 'server':
    database_path = '/work/smg/v-nicolas/level-design_raise_100/'
  else:
    database_path = '/home/nicolas/Database/level-design_raise_100/'

  image_size = 100
  nb_train_batch = 0
  nb_test_batch = 80
  nb_validation_batch = 40

  clf = Model(database_path, image_size, nbins = 13,
              batch_size = 50, histograms = False, stats = False, 
              using_GPU = using_GPU)

  # clf.show_histogram()

  clf.train(nb_train_batch = nb_train_batch,
            nb_test_batch = nb_test_batch, 
            nb_validation_batch = nb_validation_batch)

  # clf.lda_training(nb_train_batch = 800, nb_test_batch = 80)

  if config == 'server':
    test_data_path = '/work/smg/v-nicolas/level-design_raise_650/test/'
  else: 
    test_data_path = '/home/nicolas/Database/level-design_raise/test/'

  clf.test_total_images(test_data_path = test_data_path,
                        nb_images = 720, decision_rule = 'weighted_vote',
                        show_images = False, 
                        save_images = False,
                        only_green = True)

  if config == 'server':
    splicing_data_path = '/work/smg/v-nicolas/splicing/'
  else: 
    splicing_data_path = '/home/nicolas/Database/splicing/'

  clf.test_splicing(data_path = splicing_data_path, 
                    nb_images = 50,
                    minibatch_size = 25,
                    show_images = False,
                    save_images = True)
