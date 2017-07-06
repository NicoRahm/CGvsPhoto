"""
    The ``model`` module
    ======================
 
    Contains the class Model which implements the core model for CG detection, 
    training, testing and visualization functions.
"""

import os

import time
import random
from . import image_loader as il
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import csv
import configparser

import numpy as np

from PIL import Image

GPU = '/gpu:0'
config = 'server'

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score as acc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# seed initialisation
print("\n   random initialisation ...")
random_seed = int(time.time() % 10000 ) 
random.seed(random_seed)  # for reproducibility
print('   random seed =', random_seed)

# tool functions

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


def image_summaries(var, name):
  tf.summary.image(name + '_1', var[:,:,:,0:1], max_outputs = 1)
  tf.summary.image(name + '_2', var[:,:,:,1:2], max_outputs = 1)
  tf.summary.image(name + '_3', var[:,:,:,2:3], max_outputs = 1)
  # tf.summary.image(name + '_4', var[:,:,:,3:4], max_outputs = 1)
  # tf.summary.image(name + '_5', var[:,:,:,4:5], max_outputs = 1)
  # tf.summary.image(name + '_6', var[:,:,:,5:6], max_outputs = 1)
  # tf.summary.image(name + '_7', var[:,:,:,6:7], max_outputs = 1)
  # tf.summary.image(name + '_8', var[:,:,:,7:8], max_outputs = 1)

def filter_summary(filters, name):
  tf.summary.image(name + '_1', tf.stack([filters[:,:,0:1,0]]), max_outputs = 1)
  tf.summary.image(name + '_2', tf.stack([filters[:,:,1:2,0]]), max_outputs = 1)
  tf.summary.image(name + '_3', tf.stack([filters[:,:,2:3,0]]), max_outputs = 1)
  # tf.summary.image(name + '_4', tf.stack([filters[:,:,3:4,0]]), max_outputs = 1)
  # tf.summary.image(name + '_5', tf.stack([filters[:,:,0,4:5]]), max_outputs = 1)
  # tf.summary.image(name + '_6', tf.stack([filters[:,:,0,5:6]]), max_outputs = 1)
  # tf.summary.image(name + '_7', tf.stack([filters[:,:,0,6:7]]), max_outputs = 1)
  # tf.summary.image(name + '_8', tf.stack([filters[:,:,0,7:8]]), max_outputs = 1)



def weight_variable(shape, seed = None):
  """Creates and initializes (truncated normal distribution) a variable weight Tensor with a defined shape"""
  initial = tf.truncated_normal(shape, stddev=0.5, seed = random_seed)
  return tf.Variable(initial)

def bias_variable(shape):
  """Creates and initializes (truncated normal distribution with 0.5 mean) a variable bias Tensor with a defined shape"""
  initial = tf.truncated_normal(shape, mean = 0.5, stddev=0.1, seed = random_seed)
  return tf.Variable(initial)
  
def conv2d(x, W):
  """Returns the 2D convolution between input x and the kernel W"""  
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
  """Returns the result of max-pooling on input x with a 2x2 window""" 
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
  """Returns the result of average-pooling on input x with a 2x2 window""" 
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

def max_pool_10x10(x):
  """Returns the result of max-pooling on input x with a 10x10 window""" 
  return tf.nn.max_pool(x, ksize=[1, 10, 10, 1],
                           strides=[1, 10, 10, 1], padding='SAME')

def avg_pool_10x10(x):
  """Returns the result of average-pooling on input x with a 10x10 window""" 
  return tf.nn.avg_pool(x, ksize=[1, 10, 10, 1],
                           strides=[1, 10, 10, 1], padding='SAME')

def histogram(x, nbins):
  """Returns the Tensor containing the nbins values of the normalized histogram of x""" 
  h = tf.histogram_fixed_width(x, value_range = [-1.0,1.0], 
                               nbins = nbins, dtype = tf.float32)
  return(h)

def gaussian_func(mu, x, n, sigma):
  """Returns the average of x composed with a gaussian function

    :param mu: The mean of the gaussian function
    :param x: Input values 
    :param n: Number of input values
    :param sigma: Variance of the gaussian function
    :type mu: float
    :type x: Tensor
    :type n: int 
    :type sigma: float
  """ 
  gauss = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
  # return(tf.reduce_sum(gauss.pdf(xmax - tf.nn.relu(xmax - x))/n))
  return(tf.reduce_sum(gauss.pdf(x)/n))



def gaussian_kernel(x, nbins = 8, values_range = [0, 1], sigma = 0.1,image_size = 100):
  """Returns the values of x's nbins gaussian histogram 

    :param x: Input values (supposed to be images)
    :param nbins: Number of bins (different gaussian kernels)
    :param values_range: The range of the x values
    :param sigma: Variance of the gaussian functions
    :param image_size: The size of the images x (for normalization)
    :type x: Tensor
    :type nbins: int 
    :type values_range: table
    :type sigma: float
    :type image_size: int
  """ 
  mu_list = np.float32(np.linspace(values_range[0], values_range[1], nbins + 1))
  n = np.float32(image_size**2)
  function_to_map = lambda m : gaussian_func(m, x, n, sigma)
  return(tf.map_fn(function_to_map, mu_list))

def plot_gaussian_kernel(nbins = 8, values_range = [0, 1], sigma = 0.1):
  """Plots the gaussian kernels used for estimating the histogram"""

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


def classic_histogram_gaussian(x, k, nbins = 8, values_range = [0, 1], sigma = 0.6):
  """Computes gaussian histogram values for k input images"""
  function_to_map = lambda y: tf.stack([gaussian_kernel(y[:,:,i], nbins, values_range, sigma) for i in range(k)])
  res = tf.map_fn(function_to_map, x)
  return(res)

def stat(x):
  """Computes statistical features for an image x : mean, min, max and variance"""
  # sigma = tf.reduce_mean((x - tf.reduce_mean(x))**2)
  return(tf.stack([tf.reduce_mean(x), tf.reduce_min(x), tf.reduce_max(x), tf.reduce_mean((x - tf.reduce_mean(x))**2)]))

def compute_stat(x, k):
  """Computes statistical features for k images"""
  # function_to_map = lambda y: tf.stack([stat(y[:,:,i]) for i in range(k)])
  # res = tf.map_fn(function_to_map, x)
  res = tf.transpose(tf.stack([tf.reduce_mean(x, axis=[1,2]), tf.reduce_min(x, axis=[1,2]), tf.reduce_max(x, axis=[1,2]), tf.reduce_mean((x - tf.reduce_mean(x, axis=[1,2], keep_dims = True))**2, axis=[1,2])]), [1,2,0])
  return(res)

class Model:

  """
    Class Model
    ======================
 
    Defines a model for single-image CG detection and numerous methods to : 
    - Create the TensorFlow graph of the model
    - Train the model on a specific database
    - Reload past weights 
    - Test the model (simple classification, full-size images with boosting and splicing)
    - Visualize some images and probability maps
"""

  def __init__(self, database_path, image_size, config = 'Personal', filters = [32, 64],
              feature_extractor = 'Stats', remove_context = False, 
              nbins = 10, batch_size = 50, using_GPU = False):
    """Defines a model for single-image classification

    :param database_path: Absolute path to the default patch database (training, validation and testings are performed on this database)
    :param image_size: Size of the patches supposed squared
    :param config: Name of the section to use in the config.ini file for configuring directory paths (weights, training summaries and visualization dumping)
    :param filters: Table with the number of output filters of each layer
    :param feature_extractor: Two choices 'Stats' or 'Hist' for the feature extractor
    :param nbins: Number of bins on the histograms. Used only if the feature_extractor parameter is 'Hist'
    :param batch_size: The size of the batch for training
    :param using_GPU: Whether to use GPU for computation or not 
    
    :type database_path: str
    :type image_size: int
    :type config: str
    :type filters: table
    :type feature_extractor: str
    :type nbins: int
    :type batch_size: int
    :type using_GPU: bool
  """ 
    clear = lambda: os.system('clear')
    clear()
    print('   tensorFlow version: ', tf.__version__)
    
    # read the configuration file
    conf = configparser.ConfigParser()
    conf.read('config.ini')

    if config not in conf:
      raise ValueError(config + ' is not in the config.ini file... Please create the corresponding section')
    
    self.dir_ckpt = conf[config]['dir_ckpt']
    self.dir_summaries = conf[config]['dir_summaries']
    self.dir_visualization = conf[config]['dir_visualization']
    print('   Check-points directory : ' + self.dir_ckpt)
    print('   Summaries directory : ' + self.dir_summaries)
    print('   Visualizations directory : ' + self.dir_visualization)

    # setting the parameters of the model
    self.nf = filters
    self.nl = len(self.nf)
    self.filter_size = 3

    self.feature_extractor = 'Stats'

    if self.feature_extractor != 'Stats' and self.feature_extractor != 'Hist':
      raise ValueError('''Feature extractor must be 'Stats' or 'Hist' ''')

    self.database_path = database_path
    self.image_size = image_size
    self.batch_size = batch_size
    self.nbins = nbins
    self.using_GPU = using_GPU
    self.remove_context = remove_context

    # getting the database
    self.import_database()

    # create the TensorFlow graph
    if using_GPU:
      with tf.device(GPU):
        self.create_graph(nb_class = self.nb_class, 
                          feature_extractor = self.feature_extractor,
                          nl = self.nl, nf = self.nf, filter_size = self.filter_size)
    else: 
      self.create_graph(nb_class = self.nb_class, 
                        feature_extractor = self.feature_extractor,
                        nl = self.nl, nf = self.nf, filter_size = self.filter_size)



  def import_database(self): 
    """Creates a Database_loader to load images from the distant database"""

    # load data
    print('   import data : image_size = ' + 
        str(self.image_size) + 'x' + str(self.image_size) + '...')
    self.data = il.Database_loader(self.database_path, self.image_size, 
                                   proportion = 1, only_green=True)
    self.nb_class = self.data.nb_class

  def create_graph(self, nb_class, nl = 2, nf = [32, 64], filter_size = 3,
                   feature_extractor = 'Stats'): 
    """Creates the TensorFlow graph"""

    print('   create model ...')
    # input layer. One entry is a float size x size, 3-channels image. 
    # None means that the number of such vector can be of any lenght.

    if feature_extractor == 'Hist': 
      print('   Model with histograms.')

    else: 
      print('   Model with statistics.')

    graph = tf.Graph()

    with graph.as_default():

      with tf.name_scope('Input_Data'):
        x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1])
        self.x = x
        # reshape the input data:
        x_image = tf.reshape(x, [-1,self.image_size, self.image_size, 1])
        with tf.name_scope('Image_Visualization'):
          tf.summary.image('Input_Data', x_image)
        

      # first conv net layer
      print('   Creating layer 1 - Shape : ' + str(self.filter_size) + 'x' + 
            str(self.filter_size) + 'x1x' + str(nf[0]))

      with tf.name_scope('Conv1'):

        with tf.name_scope('Weights'):
          W_conv1 = weight_variable([self.filter_size, self.filter_size, 1, nf[0]], seed = random_seed)
          self.W_conv1 = W_conv1
        with tf.name_scope('Bias'):
          b_conv1 = bias_variable([nf[0]])


        # relu on the conv layer
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, 
                             name = 'Activated_1')
        self.h_conv1 = h_conv1

      self.W_convs = [W_conv1]
      self.b_convs = [b_conv1]
      self.h_convs = [h_conv1]

      image_summaries(h_conv1, 'hconv1_intra')
      filter_summary(W_conv1, 'Wconv1_intra')

      for i in range(1, nl):
        print('   Creating layer ' + str(i+1) + ' - Shape : ' + str(self.filter_size) + 'x' + 
            str(self.filter_size) + 'x' + str(nf[i-1]) + 'x' + str(nf[i]))
        # other conv 
        with tf.name_scope('Conv' + str(i+1)):
          with tf.name_scope('Weights'):
            W_conv2 = weight_variable([self.filter_size, self.filter_size, nf[i-1], nf[i]])
            self.W_convs.append(W_conv2)
          with tf.name_scope('Bias'):
            b_conv2 = bias_variable([nf[i]])
            self.b_convs.append(b_conv2)

          h_conv2 = tf.nn.relu(conv2d(self.h_convs[i-1], W_conv2) + b_conv2, 
                               name = 'Activated_2')

          self.h_convs.append(h_conv2)    


      print('   Creating feature extraction layer')
      nb_filters = nf[nl-1]
      if self.feature_extractor == 'Hist':
        # Histograms
        nbins = self.nbins
        size_flat = (nbins + 1)*nb_filters

        range_hist = [0,1]
        sigma = 0.07

        # plot_gaussian_kernel(nbins = nbins, values_range = range_hist, sigma = sigma)

        with tf.name_scope('Gaussian_Histogram'): 
          hist = classic_histogram_gaussian(self.h_convs[nl-1], k = nb_filters, 
                                            nbins = nbins, 
                                            values_range = range_hist, 
                                            sigma = sigma)
          self.hist = hist

        flatten = tf.reshape(hist, [-1, size_flat], name = "Flatten_Hist")
        self.flatten = flatten

      else: 
        nb_stats = 4
        size_flat = nb_filters*nb_stats
        with tf.name_scope('Simple_statistics'): 
          s = compute_stat(self.h_convs[nl-1], nb_filters)
          self.stat = s
          
        flatten = tf.reshape(s, [-1, size_flat], name = "Flattened_Stat")
        self.flatten = flatten


      print('   Creating MLP ')
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

      # with tf.name_scope('enforce_constraints'):
      if self.remove_context:
        # self.zero_op = tf.assign(ref = self.W_convs[0][1,1,0,:], value = tf.zeros([nf[0]]))
        self.zero_op = tf.scatter_nd_update(ref = self.W_convs[0], indices = tf.constant([[1,1,0,i] for i in range(nf[0])]), updates = tf.zeros(nf[0]))
        self.norm_op = tf.assign(ref = self.W_convs[0], value = tf.divide(self.W_convs[0],tf.reduce_sum(self.W_convs[0], axis = 3, keep_dims = True)))

      self.train_step = train_step
      print('   test ...')
      # 'correct_prediction' is a function. argmax(y, 1), here 1 is for the axis number 1
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

      # 'accuracy' is a function: cast the boolean prediction to float and average them
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

      self.accuracy = accuracy

    self.graph = graph
    print('   model created.')

  def validation_testing(self, it, nb_iterations = 20, batch_size = 50,
                         plot_histograms = False, range_hist = [0.,1.], 
                         selected_hist_nb = 8, run_name = '',
                         show_filters = True):
    """Computes validation accuracy during training and plots some visualization.
      
    Returns the accuracy on the validation data. Can also plot some histograms of the filtered images 
    (if the Hist layer is selected) and the first layer's filters.

    :param it: The number of the iteration in the training process
    :param nb_iterations: The number of batches to process on the validation set
    :param batch_size: Batch size when loading the validation images
    :param plot_hitograms: Whether to plot the histograms or not
    :param range_hist: The value range for plotting the histograms
    :param selected_hist_nb: The number of histograms to plot
    :param run_name: The name of the training run
    :param show_filters: Whether to show the first layer's filters
    :type it: int
    :type nb_iterations: int
    :type batch_size: int
    :type plot_hitograms: bool
    :type range_hist: table
    :type selected_hist_nb: int
    :type run_name: str
    :type show_filters: bool
    """
    if show_filters: 
      
      nb_height = 4
      nb_width = int(self.nb_conv1/nb_height)

      img, axes = plt.subplots(nrows = nb_width, ncols = nb_height)
      gs1 = gridspec.GridSpec(nb_height, nb_width)
      for i in range(self.nb_conv1):
        ax1 = plt.subplot(gs1[i])
        ax1.axis('off')
        im = plt.imshow(self.W_conv1[:,:,0,i].eval(), cmap = 'jet')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([]) 
        # axes.get_yaxis().set_ticks([])
        # plt.ylabel('Kernel ' + str(i), fontsize = 5.0)
        # ax1.set_ylabel('Kernel ' + str(i), fontsize = 5.0)
        ax1.set_title("Filter " + str(i + 1), fontsize = 12.0)    

      img.subplots_adjust(wspace = 0.1, hspace = 0.6, right = 0.7)
      cbar_ax = img.add_axes([0.75, 0.15, 0.03, 0.7])
      cbar = img.colorbar(im, ticks=[-0.5, 0, 0.5], cax=cbar_ax)
      cbar.ax.set_yticklabels(['< -0.5', '0', '> 0.5'])
      plt.show(img)
      plt.close()     

    if plot_histograms and self.feature_extractor != 'Hist':
      print("Can't plot the histograms, feature extractor is 'Stats'...")

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

      
      if plot_histograms and self.feature_extractor == 'Hist':
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

    if plot_histograms and self.feature_extractor == 'Hist':
      # Plotting mean histogram for CGG
      fig = plt.figure(1)
      for k in range(selected_hist_nb):
        plt.subplot(selected_hist_nb/2, 2, k+1)
        plt.bar(np.linspace(range_hist[0], range_hist[1], self.nbins+1), 
                            hist_CGG[k], width = 1/(self.nbins + 1))
        plt.plot(np.linspace(range_hist[0], range_hist[1], self.nbins+1), 
                             hist_CGG[k], 'r')
        fig.suptitle("Mean histogram for CGG", fontsize=14)
      plt.show()
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
      plt.show()
      plt.close()



    validation_accuracy /= nb_iterations
    print("     step %d, training accuracy %g (%d validations tests)"%(it, validation_accuracy, validation_batch_size*nb_iterations))
    return(validation_accuracy)


  def train(self, nb_train_batch, nb_test_batch, 
            nb_validation_batch, validation_frequency = 10, show_filters = False):
    """Trains the model on the selected database training set.
      
    Trains a blank single-image classifer (or initialized with some pre-trained weights). 
    The weights are saved in the corresponding file along training, validation is computed, 
    showed and saved at the end. Finnaly, summaries are generated. 
    Testing is also performed for single-images.

    :param nb_train_batch: The number of batches to train (can be on multiple epochs)
    :param nb_test_batch: The number of batches to test
    :param nb_validation_batch: The number of batch for validation
    :param validation_frequency: Performs validation testing every validation_frequency batches
    :param show_filters: Whether to show the first layer's filters at each validation step
    :type nb_train_batch: int
    :type nb_test_batch: int
    :type nb_validation_batch: int
    :type validation_frequency: int
    :type show_filters: bool
    """
    run_name = input("   Choose a name for the run : ")
    path_save = self.dir_ckpt + run_name
    acc_name = self.dir_summaries + run_name + "/validation_accuracy_" + run_name + ".csv"


    # computation time tick
    start_clock = time.clock()
    start_time = time.time()
    batch_clock = None

    # start a session
    print('   start session ...')
    with tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=self.using_GPU)) as sess:

      merged = tf.summary.merge_all()
      
      if not os.path.exists(self.dir_summaries + run_name):
        os.mkdir(self.dir_summaries + run_name)


      train_writer = tf.summary.FileWriter(self.dir_summaries + run_name,
                                           sess.graph)

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      saver = tf.train.Saver()
      print('   variable initialization ...')

      restore_weigths = input("\nRestore weight from previous session ? (y/N) : ")

      if restore_weigths == 'y':
        file_to_restore = input("\nName of the file to restore (Directory : " + 
                                self.dir_ckpt + ') : ')
        saver.restore(sess, self.dir_ckpt + file_to_restore)
        print('\n   Model restored\n')
        

      # Train
      print('   train ...')
      start_clock = time.clock()
      start_time = time.time()
      validation_accuracy = []
      for i in range(nb_train_batch):

          # enforce constraints on first layer : 
          if self.remove_context: 
            sess.run(self.zero_op)
            # sess.run(self.norm_op)
        
          # evry validation_frequency batches, test the accuracy
          if i%validation_frequency == 0 :
              
              if i%100 == 0:
                plot_histograms = False
              else:
                plot_histograms = False

              v = self.validation_testing(i, nb_iterations = nb_validation_batch, 
                                      batch_size = self.batch_size, 
                                      plot_histograms = plot_histograms,
                                      run_name = run_name,
                                      show_filters = show_filters)
              validation_accuracy.append(v)
              
          # regular training



          batch = self.data.get_next_train_batch(self.batch_size, False, True, True)
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


    # final test
      print('   final test ...')
      test_accuracy = 0
      # test_auc = 0
      nb_iterations = 20
      self.data.test_iterator = 0
      for _ in range( nb_iterations ) :
          batch_test = self.data.get_batch_test(self.batch_size, False, True, True)
          feed_dict = {self.x:batch_test[0], self.y_: batch_test[1], self.keep_prob: 1.0}
          test_accuracy += self.accuracy.eval(feed_dict)
          # test_auc += sess.run(auc, feed_dict)[0]

                
      test_accuracy /= nb_iterations
      print("   test accuracy %g"%test_accuracy)

      # test_auc /= (nb_iterations - 1)
      # print("   test AUC %g"%test_auc)
      if nb_train_batch > validation_frequency:
        plt.figure()
        plt.plot(np.linspace(0,nb_train_batch,int(nb_train_batch/10)), validation_accuracy)
        plt.title("Validation accuracy during training")
        plt.xlabel("Training batch")
        plt.ylabel("Validation accuracy")
        plt.show()
        plt.close()

    # done
    print("   computation time (cpu) :",time.strftime("%H:%M:%S", time.gmtime(time.clock()-start_clock)))
    print("   computation time (real):",time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
    print('   done.')


  def show_histogram(self):
    """Plots histograms of the last layer outputs for some images"""

    with tf.Session(graph=self.graph) as sess:

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      saver = tf.train.Saver()
      print('   variable initialization ...')

      file_to_restore = input("\nName of the file to restore (Directory : " + 
                                self.dir_ckpt + ') : ')
      saver.restore(sess, self.dir_ckpt + file_to_restore)
      print('\n   Model restored\n')

      batch = self.data.get_next_train_batch(self.batch_size, False, True, True)
      feed_dict = {self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0}
      conv = self.h_conv2.eval(feed_dict = feed_dict)

      for i in range(self.batch_size):
        plt.figure()

        plt.hist(np.reshape(conv[i,:,:,0], (self.image_size*self.image_size,)))
        plt.show()

  def mean_histogram(self, nb_images = 5000):
    print("   Showing the histograms of filtered images...")
    with tf.Session(graph=self.graph) as sess:

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      saver = tf.train.Saver()
      print('   variable initialization ...')

      file_to_restore = input("\nName of the file to restore (Directory : " + 
                                  self.dir_ckpt + ') : ')
      saver.restore(sess, self.dir_ckpt + file_to_restore)
      print('\n   Model restored\n')
      j = 0
      nreal = 0
      ncgg = 0
      while j < nb_images:

        batch = self.data.get_next_train_batch(self.batch_size, False, True, True)
        feed_dict = {self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0}
        conv = self.h_conv1.eval(feed_dict = feed_dict)

        nbins = 150
        hist_values_CGG = np.zeros((nbins,))
        hist_values_Real = np.zeros((nbins,))

        for i in range(self.batch_size):
          if batch[1][i][0] == 1:
            # print(conv[i,:,:,15])
            hist_values_Real += np.histogram(conv[i,:,:,1], bins = nbins, range = (0., 1.))[0]
            nreal += 1

          else:
            # print(conv[i,:,:,15])
            hist_values_CGG += np.histogram(conv[i,:,:,1], bins = nbins, range = (0., 1.))[0]
            ncgg += 1

        j+= self.batch_size

      hist_values_CGG /= ncgg
      hist_values_Real /= nreal

      plt.figure()
      plt.plot(np.linspace(0,1, nbins), hist_values_Real, color = 'b', 
                 label = 'Real')
      plt.plot(np.linspace(0,1, nbins), hist_values_CGG, color = 'r', 
                 label = 'CGG')
      plt.legend()
      plt.show()

  def lda_training(self, nb_train_batch, nb_test_batch):
    """Trains a LDA classifier on top of the feature extractor.
      
    Restores the weights of the feature extractor and trains a new LDA classifier. The trained LDA can then be reused.
    Finally tests the pipeline on the test dataset. 

    :param nb_train_batch: The number of batches to train (can be on multiple epochs)
    :param nb_test_batch: The number of batches to test
    :type nb_train_batch: int
    :type nb_test_batch: int
    """
    self.lda_classifier = LinearDiscriminantAnalysis()

    # start a session
    print('   start session ...')
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      print('   variable initialization ...')
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = input("\nName of the file to restore (Directory : " + 
                              self.dir_ckpt + ') : ')
      saver.restore(sess, self.dir_ckpt + file_to_restore)

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
      self.clf = self.lda_classifier

  def svm_training(self, nb_train_batch, nb_test_batch):
    """Trains a SVM classifier (RBF kernel) on top of the feature extractor.
      
    Restores the weights of the feature extractor and trains a new SVM classifier with RBF kernel. The trained SVM can then be reused.
    Finally tests the pipeline on the test dataset. 

    :param nb_train_batch: The number of batches to train (can be on multiple epochs)
    :param nb_test_batch: The number of batches to test
    :type nb_train_batch: int
    :type nb_test_batch: int
    """

    self.svm_classifier = SVC(probability = True)
    # start a session
    print('   start session ...')
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      print('   variable initialization ...')
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = input("\nName of the file to restore (Directory : " + 
                              self.dir_ckpt + ') : ')
      saver.restore(sess, self.dir_ckpt + file_to_restore)

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
      self.clf = self.svm_classifier


  def test_total_images(self, test_data_path, nb_images, 
                        minibatch_size = 25, decision_rule = 'majority_vote',
                        show_images = False,
                        save_images = False,
                        only_green = True, other_clf = False): 
    """Performs boosting for classifying full-size images.

    Decomposes each image into patches (with size = self.image_size), computes the posterior probability of each class
    and uses a decision rule to classify the full-size image.
    Optionnaly plots or save the probability map and the original image in the visualization directory.

    :param test_data_path: The absolute path to the test dataset. Must contain two directories : CGG/ and Real/
    :param nb_images: The number of images to test
    :param minibatch_size: The size of the batch to process the patches
    :param decision_rule: The decision rule to use to aggregate patches prediction
    :param show_images: Whether to show images or not 
    :param save_images: Whether to save images or not
    :param only_green: Whether to take only the green channel of the image
    :param other_clf: Whether to use aother classifier (LDA or SVM). If True, takes the lastly trained

    :type test_data_path: str
    :type nb_images: int
    :type minibatch_size: int
    :type decision_rule: str
    :type show_images: bool
    :type save_images: bool
    :type only_green: bool
    :type other_clf:bool
    """
    valid_decision_rule = ['majority_vote', 'weighted_vote']
    if decision_rule not in valid_decision_rule:
      raise NameError(decision_rule + ' is not a valid decision rule.')
    test_name = input("   Choose a name for the test : ")
    if(save_images):
      if not os.path.exists(self.dir_visualization + test_name):
        os.mkdir(self.dir_visualization + test_name)

    if not only_green:
      print('   No visualization when testing all channels...')
      show_images = False
      save_images = False
    print('   Testing for the database : ' + test_data_path)
    print('   start session ...')
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      print('   variable initialization ...')
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = input("\nName of the file to restore (Directory : " + 
                              self.dir_ckpt + ') : ')
      saver.restore(sess, self.dir_ckpt + file_to_restore)

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
        nb_im = 0
        while j < batch_size:
          if other_clf:
            feed_dict = {self.x: batch[j:j+minibatch_size], self.keep_prob: 1.0}
            features = self.flatten.eval(feed_dict = feed_dict)
            pred = np.log(self.clf.predict_proba(features) + 0.00001)


          else:
            feed_dict = {self.x: batch[j:j+minibatch_size], self.keep_prob: 1.0}
            pred = self.y_conv.eval(feed_dict)
          
          nb_im += pred.shape[0]
          label_image = np.argmax(pred, 1)
          d =  np.max(pred, 1) - np.min(pred, 1)
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
          print(prediction/nb_im)
          scores.append(prediction/nb_im)

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
          self.image_visualization(path_save = self.dir_visualization + test_name, 
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
      fpr, tpr, thresholds = roc_curve(np.array(y), 0.5 + np.array(scores)/10)

      print(0.5 + np.array(scores)/np.max(np.array(scores)))
      print(thresholds)

      filename = '/home/nicolas/Documents/ROC/' + test_name + '.csv'
      print('Saving tpr and fpr in file : ' + filename)
      with open(filename, 'w') as file:
        try:
          writer = csv.writer(file)
       
          for i in range(fpr.shape[0]):
            writer.writerow([str(fpr[i]), str(tpr[i])])
          print('   done.')
        finally:
          file.close()

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
    """Computes image visualization and save/show it

    Permits to visualize the probability map of the image. Green color represents correctly classified patches
    and red wrongly classified ones. The intensity depends on the level of certainty.

    :param path_save: The absolute path where images should be saved
    :param file_name: The name of input image file
    :param images: An array containing patches extracted from the full-size image
    :param width: The width of the full-size image
    :param height: The height of the full-size image
    :param diff: Differences between log posterior probabilities for each patch
    :param original: The original image
    :param show_images: Whether to show images or not 
    :param save_images: Whether to save images or not
    :param prob_map: Whether to save the probability map
    :param save_original: Whether to save the original image

    :type path_save: str
    :type file_name: str
    :type images: numpy array
    :type width: int
    :type height: int
    :type diff: numpy array
    :type original: numpy array
    :type show_images: bool
    :type save_images: bool
    :type prob_map: bool
    :type save_original: bool
    """
    nb_width = int(width/self.image_size)
    nb_height = int(height/self.image_size)
    m = 10
    img = plt.figure(figsize = (nb_width, nb_height))
    
    gs1 = gridspec.GridSpec(nb_height, nb_width)
    
    
    for i in range(len(images)):

      cdict_green = {'red': ((0.0,0.0,0.0),
                             (1.0,1.0 - diff[i]/m,1.0 - diff[i]/m)),
                     'blue': ((0.0,0.0,0.0),
                              (1.0,1.0 - diff[i]/m,1.0 - diff[i]/m)),
                     'green': ((0.0,0.0,0.0),
                               (1.0,1.0,1.0))}

      cdict_red = {'red': ((0.0,0.0,0.0),
                               (1.0,1.0,1.0)),
                    'blue': ((0.0,0.0,0.0),
                                (1.0,1.0 - diff[i]/m,1.0 - diff[i]/m)),
                    'green': ((0.0,0.0,0.0),
                                 (1.0,1.0 - diff[i]/m,1.0 - diff[i]/m))}
      

      ax1 = plt.subplot(gs1[i])
      ax1.axis('off')
      if labels_pred[i] == 'Real':
        if diff[i] > 0.4:
          cmap = mcolors.LinearSegmentedColormap('my_green', cdict_green, 100)
        else:
          cmap = 'gray'
      else: 
        if diff[i] > 0.4:
          cmap = mcolors.LinearSegmentedColormap('my_red', cdict_red, 100)
        else:
          cmap = 'gray'

      images[i,0,0,0] = 0
      images[i,0,1,0] = 1
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
                             (1.0,1.0 - diff[i]/m,1.0 - diff[i]/m)),
                     'blue': ((0.0,0.0,0.0),
                              (1.0,1.0 - diff[i]/m,1.0 - diff[i]/m)),
                     'green': ((0.0,0.0,0.0),
                               (1.0,1.0,1.0))}

          cdict_red = {'red': ((0.0,0.0,0.0),
                                   (1.0,1.0,1.0)),
                        'blue': ((0.0,0.0,0.0),
                                    (1.0,1.0 - diff[i]/m,1.0 - diff[i]/m)),
                        'green': ((0.0,0.0,0.0),
                                     (1.0,1.0 - diff[i]/m,1.0 - diff[i]/m))}
          
          ax1 = plt.subplot(gs1[i])
          ax1.axis('off')
          if labels_pred[i] == true_label:
            if diff[i] > 0.4:
              cmap = mcolors.LinearSegmentedColormap('my_green', cdict_green, 100)
            else:
              cmap = 'gray'
              map_im = map_im*0.7
          else: 
            if diff[i] > 0.4:
              cmap = mcolors.LinearSegmentedColormap('my_red', cdict_red, 100)
            else:
              cmap = 'gray'
              map_im = map_im*0.7

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

  def show_filtered(self, image_file):
    print('   Loading image from file : ' + image_file)
    im = Image.open(image_file)
    im = np.reshape(np.array([np.asarray(im)]), (1,self.image_size, self.image_size, 1))

    print('   start session ...') 
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      print('   variable initialization ...')
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = input("\nName of the file to restore (Directory : " + 
                              self.dir_ckpt + ') : ')
      saver.restore(sess, self.dir_ckpt + file_to_restore)

      feed_dict = {self.x: im, self.keep_prob: 1.0}
      filtered = self.h_conv1.eval(feed_dict = feed_dict)

      for i in range(filtered.shape[3]):
        plt.figure()
        plt.imshow(filtered[0,:,:,i], cmap = 'gray')
        plt.show()



  def test_splicing(self, data_path, nb_images, save_images = True, show_images = False,
                    minibatch_size = 25):
    """Computes image visualization for spliced images

    Decomposes each image into patches (with size = self.image_size), computes the posterior probability of each class
    and show the probability map.

    :param data_path: Path to the spliced images. Should contain two directories : CGG/ and Real/
    :param nb_images: Number of spliced images to process
    :param show_images: Whether to show images or not 
    :param save_images: Whether to save images or not
    :param minibatch_size: The size of the batch to process the patches


    :type data_path: str
    :type nb_images: int
    :type show_images: bool
    :type save_images: bool
    :type minibatch_size: int
    """
    if(save_images):
      test_name = input("   Choose a name for the test : ")
      path_save = self.dir_visualization + test_name
      if not os.path.exists(self.dir_visualization + test_name):
        os.mkdir(self.dir_visualization + test_name)

    else: 
      path_save = ''

    print('   start session ...')
    with tf.Session(graph=self.graph) as sess:
      saver = tf.train.Saver()
      print('   variable initialization ...')
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      file_to_restore = input("\nName of the file to restore (Directory : " + 
                              self.dir_ckpt + ') : ')
      saver.restore(sess, self.dir_ckpt + file_to_restore)

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
  nb_train_batch = 5000
  nb_test_batch = 80
  nb_validation_batch = 40

  clf = Model(database_path, image_size, nbins = 11,
              batch_size = 50, histograms = False, stats = True, 
              using_GPU = using_GPU)

  # clf.mean_histogram()

  # clf.show_filtered('/home/nicolas/Database/level-design_dresden_100/train/CGG/train153.jpg')

  clf.train(nb_train_batch = nb_train_batch,
            nb_test_batch = nb_test_batch, 
            nb_validation_batch = nb_validation_batch,
            save_filters = False)

  # clf.svm_training(nb_train_batch = 800, nb_test_batch = 80)


  if config == 'server':
    test_data_path = '/work/smg/v-nicolas/level-design_raise_650/test/'
  else: 
    test_data_path = '/home/nicolas/Database/level-design_raise_650/test/'

  clf.test_total_images(test_data_path = test_data_path,
                        nb_images = 720, decision_rule = 'weighted_vote',
                        show_images = False, 
                        save_images = False,
                        only_green = True,
                        other_clf = False)


  if config == 'server':
    test_data_path = '/work/smg/v-nicolas/level-design_raise/test/'
  else: 
    test_data_path = '/home/nicolas/Database/level-design_raise/test/'

  clf.test_total_images(test_data_path = test_data_path,
                        nb_images = 720, decision_rule = 'weighted_vote',
                        show_images = False, 
                        save_images = False,
                        only_green = True,
                        other_clf = False)

  if config == 'server':
    splicing_data_path = '/work/smg/v-nicolas/splicing/'
  else: 
    splicing_data_path = '/home/nicolas/Database/splicing/'

  clf.test_splicing(data_path = splicing_data_path, 
                    nb_images = 50,
                    minibatch_size = 25,
                    show_images = False,
                    save_images = True)
