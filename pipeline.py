
print("   reset python interpreter ...")
import os
clear = lambda: os.system('cls')
clear()
import time
import random
import plot_history as ph
import image_loader as il
import tensorflow as tf

from PIL import Image, ImageDraw
import numpy as np

# computation time tick
start_clock = time.clock()
start_time = time.time()

# seed initialisation
print("\n   random initialisation ...")
random_seed = int(time.time() % 10000 ) 
random.seed(random_seed)  # for reproducibility
print('   random seed =', random_seed)



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
print('   import data ...')
#data = il.Database_loader('/home/nozick/Desktop/database/cg_pi_64/test5', image_size, only_green=True)
data = il.Database_loader('/home/nicolas/Documents/Stage 3A/Test', image_size, only_green=True)



print('   create model ...')
# input layer. One entry is a float size x size, 3-channels image. 
# None means that the number of such vector can be of any lenght.
x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])

# reshape the input data:
# size,size: width and height
# 1: color channels
# -1 :  ???
x_image = tf.reshape(x, [-1,image_size, image_size, 1])

# Filtering with laplacian filter
laplacian = tf.constant([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], tf.float32)
laplacian_filter = tf.reshape(laplacian, [3, 3, 1, 1])

x_image_filtered = conv2d(x_image, laplacian_filter)


# first conv net layer
# conv_matrix_width : 5
# conv_matrix_height : 5
# conv_matrix nb channel ??? : 1
# nb matrices : 32 ?



W_conv1 = weight_variable([5, 5, 1, 32], seed = random_seed)
b_conv1 = bias_variable([32])

# relu on the conv layer
h_conv1 = tf.nn.relu(conv2d(x_image_filtered, W_conv1) + b_conv1)

m_pool = max_pool_2x2(h_conv1)

# second conv 
W_conv2 = weight_variable([5, 5, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(m_pool, W_conv2) + b_conv2)



# max pool
#m_pool3 = max_pool_10x10(h_conv1)
m_pool3 = max_pool_2x2(h_conv2)

# average pool
#a_pool3 = avg_pool_10x10(h_conv1)
#a_pool3 = avg_pool_2x2(h_conv2)

# difference between max and average
#diff_1 = tf.subtract(m_pool3, a_pool3)
diff_1 = m_pool3

# flattern 
# with the 2 pooling, the image size is 7x7
# there are 64 conv matrices
h_pool2_flat = tf.reshape(diff_1, [-1, int(image_size/2)*int(image_size/2)*64])

# Densely Connected Layer
# we add a fully-connected layer with 1024 neurons 
W_fc1 = weight_variable([int(image_size/2)*int(image_size/2) * 64, 1024])
b_fc1 = bias_variable([1024])
# put a relu
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Densely Connected Layer
# we add a fully-connected layer with 1024 neurons 
W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])
# put a relu
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# dropout
keep_prob2 = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, data.nb_class])
b_fc2 = bias_variable([data.nb_class])
y_conv = tf.matmul(h_fc2_drop, W_fc2) + b_fc2

# support for the learning label
y_ = tf.placeholder(tf.float32, [None, data.nb_class])




# Define loss (cost) function and optimizer
print('   setup loss function and optimizer ...')

# softmax to have normalized class probabilities + cross-entropy
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
#print('\nsoftmax_cross_entropy shape : ', softmax_cross_entropy.get_shape() )

cross_entropy_mean = tf.reduce_mean(softmax_cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)

print('   test ...')
# 'correct_prediction' is a function. argmax(y, 1), here 1 is for the axis number 1
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# 'accuracy' is a function: cast the boolean prediction to float and average them
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# start a session
print('   start session ...')
sess = tf.InteractiveSession()

print('   variable initialization ...')
tf.global_variables_initializer().run()


# Train
print('   train ...')
history = []
for i in range(81): # in the test 20000
  
    # evry 100 batches, test the accuracy
    if i%10 == 0 :
        validation_batch_size = 10       # size of the batches
        validation_accuracy = 0
        data.validation_iterator = 0
        nb_iterations = 100
        for _ in range( nb_iterations ) :
            batch_validation = data.get_batch_validation(batch_size=validation_batch_size, random_flip_flop = True, random_rotate = True)
            feed_dict = {x:batch_validation[0], y_: batch_validation[1], keep_prob: 1.0}
            validation_accuracy += accuracy.eval(feed_dict)
          
        validation_accuracy /= nb_iterations
        print("     step %d, training accuracy %g (%d validations tests)"%(i, validation_accuracy, validation_batch_size*nb_iterations))
        history.append(validation_accuracy)
        
        
    # regular training
#    print('get batch ...')
    batch_size = 50
    batch = data.get_next_train_batch(batch_size, True, True)
#    print('train ...')
    feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
    train_step.run(feed_dict)

    
# history
print('   plot history')
with open("/tmp/history.txt", "w") as history_file:
    for item in history:
        history_file.write("%f\n" %item)

with open("./history_v2.txt", "w") as history_file:
    for item in history:
        history_file.write("%f\n" %item)
        
ph.plot_history("/tmp/history.txt")


# final test
print('   final test ...')
test_batch_size = 10       # size of the batches
test_accuracy = 0
nb_iterations = 200
data.test_iterator = 0
for _ in range( nb_iterations ) :
    batch_test = data.get_batch_test(batch_size=test_batch_size, random_flip_flop = True, random_rotate = True)
    feed_dict = {x:batch_test[0], y_: batch_test[1], keep_prob: 1.0}
    test_accuracy += accuracy.eval(feed_dict)
          
test_accuracy /= nb_iterations
print("   test accuracy %g"%test_accuracy)


#batch_test = data.get_batch_test(max_images=50)
#print("   test accuracy %g"%accuracy.eval(feed_dict={x: batch_test[0], 
#                                                     y_: batch_test[1], 
#                                                     keep_prob: 1.0}))

# done
print("   computation time (cpu) :",time.strftime("%H:%M:%S", time.gmtime(time.clock()-start_clock)))
print("   computation time (real):",time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
print('   done.')


# test some images
print('   do some test on doctored images.')
final_prediction = tf.argmax(y_conv,1)
images = il.get_image_filename_from_dir('/home/nozick/Dropbox/deepLearning/keras/cg_pi/data/test') 

for im in images :
    
    random_prefix = ''.join(random.choice('0123456789ABCDEF') for i in range(7))
    im.save('/tmp/' + random_prefix + '_image.jpg')  
    print('random prefix =',random_prefix)
       
    # for each sub image
    for i in range(0, im.size[0]-image_size, image_size) :
        for j in range(0, im.size[1]-image_size, image_size) :
            box = (i, j, i+image_size, j+image_size)
            sub_im = im.crop(box)
            sub_im = np.asarray(sub_im)
            sub_im = sub_im[:,:,1]
            sub_im = sub_im.astype(np.float32) / 255.
            sub_im = sub_im.reshape(1,image_size, image_size, 1)
            
            feed_dict = {x:sub_im, keep_prob: 1.0}
            result = final_prediction.eval(feed_dict) 
            #print('result = ', result[0])
            if (result[0] == 0) :
                draw = ImageDraw.Draw(im)
                box = (i, j, i+image_size, j+image_size)  
                draw.arc(box, 0, 360)
                # simulate thinckness = 2
                box = (i+2, j+2, i+image_size-2, j+image_size-2)  
                draw.arc(box, 0, 360)
                del draw
                
    # save the images
    im.save('/tmp/' + random_prefix + '_cnn.jpg')    

print('   done.')












 