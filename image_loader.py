#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:00:21 2016

@author: nozick
"""

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os, os.path
import sys
import random


class Database_loader :

    def __init__(self, directory, size, proportion = 1.0, seed=42, only_green=True) :

        # data init
        self.dir = directory          # directory with the train / test / validation sudirectories
        self.size = size              # size of the sub image that should be croped
        self.nb_channels = 3          # return only the green channel of the images
        self.proportion = proportion
        if(only_green == True) :
            self.nb_channels = 1
        self.file_train = []          # list of the train images : tuple (image name / class)
        self.file_test = []           # list of the test images : tuple (image name / class)
        self.file_validation = []     # list of the validation images : tuple (image name / class)
        self.image_class = []         # list of the class (label) used in the process
        self.nb_class = 0
        self.train_iterator = 0       # iterator over the train images
        self.test_iterator = 0        # iterator over the test images
        self.validation_iterator = 0  # iterator over the validation images

        self.load_images(seed)        # load the data base


    def extract_channel(self, rgb_image, channel=1) :
        if channel > 2 :
            channel = 2
        return rgb_image[:,:,channel]

    def get_immediate_subdirectories(self,a_dir) :
        # return the list of sub directories of a directory
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    def load_images_in_dir(self, dir_name, image_class) :

        # file extension accepted as image data
        proportion = self.proportion
        valid_image_extension = [".jpg",".gif",".png",".tga",".tif", ".JPG"]

        file_list = []

        for c in image_class :
            nb_image_per_class = 0
            file_list_by_class = []
            for filename in os.listdir(dir_name+'/'+c):
                # check if the file is an image
                extension = os.path.splitext(filename)[1]
                if extension.lower() in valid_image_extension:
                    file_list_by_class.append(filename)

            for i in range(int(len(file_list_by_class)*proportion)):
                file_list.append((file_list_by_class[i],c))
                nb_image_per_class += 1
            print('    ',c,nb_image_per_class,'images loaded')

        return file_list

    def load_images(self, seed) :

        # check if train / test / validation directories exists
        train_dir_name = self.dir + '/train'
        if not os.path.exists(train_dir_name):
            print("error: train directory does not exist")
            sys.exit(0)
            return

        validation_dir_name = self.dir + '/validation'
        if not os.path.exists(validation_dir_name):
            print("error: validation directory does not exist")
            sys.exit(0)
            return

        test_dir_name = self.dir + '/test'
        if not os.path.exists(test_dir_name):
            print("error: test directory does not exist")
            return []
            sys.exit(0)

        # count number of classes
        self.image_class = self.get_immediate_subdirectories(train_dir_name)
        self.nb_class = len(self.image_class)
        print('     number of classes :', self.nb_class, '   ', self.image_class)

        # load image file name and class
        print("\n     train data")
        self.file_train = self.load_images_in_dir(train_dir_name,self.image_class)
        print("\n     test data")
        self.file_test = self.load_images_in_dir(test_dir_name,self.image_class)
        print("\n     validation data")
        self.file_validation = self.load_images_in_dir(validation_dir_name,self.image_class)

        # shuffle the lists
        print("\n     shuffle lists ...")
        random.seed(seed)
        random.shuffle(self.file_train)
        random.shuffle(self.file_test)
        random.shuffle(self.file_validation)
        #print(self.file_train)

        #print("\n     loading done.")


    def get_next_train(self, crop = True, random_flip_flop = False, random_rotate = False, verbose = False) :

        # load next image (size should be big enough)
        image = []
        while True:
        
            # pop file name and class
            data = self.file_train[self.train_iterator]
            self.train_iterator += 1
            if self.train_iterator >= len(self.file_train) :
                self.train_iterator = 0

            # load image
            file_name = self.dir + '/train/' + data[1] + '/' + data[0]
            image = Image.open(file_name)
            if(verbose) :
                print("  ", file_name)
                print( '     index  :', self.train_iterator -1)
                print( '     width  :', image.size[0] )
                print( '     height :', image.size[1] )
                print( '     mode   :', image.mode    )
                print( '     format :', image.format  )

            # image size test
            if crop and (image.size[0] <= self.size or image.size[1] <= self.size) :
                if(verbose) :
                    print('image too small for cropping (train) : ', data[1] + '/' + data[0])
            else : 
                break

                
        # crop image
        if crop:
            crop_width  = random.randint(0, image.size[0]-self.size-1)
            crop_height = random.randint(0, image.size[1]-self.size-1)
            box = (crop_width, crop_height, crop_width+self.size, crop_height+self.size)
    #        print('crop ', box)
            image = image.crop(box)

            # image transform
            #image.save(self.dir+'/'+str(self.train_iterator -1)+'.jpg')
            orientation = [ Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
            flip = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]

            if (random_flip_flop == True) :
                if (random.choice([True, False])) :
                    image = image.transpose(random.choice(flip))

            if (random_rotate == True) :
                if (random.choice([True, False])) :
                    image = image.transpose(random.choice(orientation))

            #image.save(self.dir+'/tranpose_'+str(self.train_iterator -1)+'.jpg')

            # convert the image into a array
            image = np.asarray(image)
            
            # extract green
            if( self.nb_channels == 1 ) :
                image = self.extract_channel(image,1)
        else: 
            image = np.asarray(image)
        # convert to float image
        image = image.astype(np.float32) / 255.
        #image = image.reshape(1, self.size, self.size, 3)
        image = image.reshape(self.size, self.size, self.nb_channels)

        # buils class label
        label = np.zeros(len(self.image_class))
        pos = self.image_class.index(data[1])
        label[pos] = 1.0
        
        # return image and label
        return (image, label)

        
    def get_next_train_batch(self, batch_size = 50, crop = True, random_flip_flop = False, random_rotate = False) :

        batch_image = np.empty([batch_size, self.size, self.size, self.nb_channels])
        batch_label = np.empty([batch_size, self.nb_class])
        for i in range(0,batch_size) :
            data = self.get_next_train(crop, random_flip_flop,random_rotate, verbose=False)
            batch_image[i] = data[0]
            batch_label[i] = data[1]

        return (batch_image.astype(np.float32),batch_label)

        
    def get_next_test(self, crop = True, random_flip_flop = False, random_rotate = False, verbose = False) :

        
        # load next image (size should be big enough)
        image = []
        while True:
          
            # pop file name and class
            data = self.file_test[self.test_iterator]
            self.test_iterator += 1
            if self.test_iterator >= len(self.file_test) :
                self.test_iterator = 0
    
            # load image
            file_name = self.dir + '/test/' + data[1] + '/' + data[0]
            image = Image.open(file_name)
            if(verbose) :
                print("  ", file_name)
                print( '     index  :', self.train_iterator -1)
                print( '     width  :', image.size[0] )
                print( '     height :', image.size[1] )
                print( '     mode   :', image.mode    )
                print( '     format :', image.format  )

            # image size test
            if crop and (image.size[0] <= self.size or image.size[1] <= self.size) :
                if(verbose) :
                    print('image too small for cropping (test) : ', data[1] + '/' + data[0])
            else : 
                break


        # crop image
        if crop:
            crop_width  = random.randint(0, image.size[0]-self.size-1)
            crop_height = random.randint(0, image.size[1]-self.size-1)
            box = (crop_width, crop_height, crop_width+self.size, crop_height+self.size)
            image = image.crop(box)

            # image transform
            #image.save(self.dir+'/'+str(self.train_iterator -1)+'.jpg')
            orientation = [ Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
            flip = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]

            if (random_flip_flop == True) :
                if (random.choice([True, False])) :
                    image = image.transpose(random.choice(flip))

            if (random_rotate == True) :
                if (random.choice([True, False])) :
                    image = image.transpose(random.choice(orientation))


            # convert the image into a array
            image = np.asarray(image)
            
            # extract green
            if( self.nb_channels == 1 ) :
                image = self.extract_channel(image,1)
        else: 
            image = np.asarray(image)
        # convert to float image
        image = image.astype(np.float32) / 255.
        #image = image.reshape(1, self.size, self.size, 3)
        image = image.reshape(self.size, self.size, self.nb_channels)
        
        # buils class label
        label = np.zeros(len(self.image_class))
        pos = self.image_class.index(data[1])
        label[pos] = 1.0
        
        # return image and label
        return (image, label)

        
    def get_batch_test(self, batch_size = 50, crop = True, random_flip_flop = False, random_rotate = False) :
        
        batch_image = np.empty([batch_size, self.size, self.size, self.nb_channels])
        batch_label = np.empty([batch_size, self.nb_class])
        for i in range(0,batch_size) :
            data = self.get_next_test(crop, random_flip_flop,random_rotate)
            batch_image[i] = data[0]
            batch_label[i] = data[1]

        return (batch_image.astype(np.float32),batch_label)
      

    def get_next_validation(self, crop = True, random_flip_flop = False, random_rotate = False, verbose = False) :

        
        # load next image (size should be big enough)
        image = []
        while True:

            # pop file name and class
            data = self.file_validation[self.validation_iterator]
            self.validation_iterator += 1
            if self.validation_iterator >= len(self.file_validation) :
                self.validation_iterator = 0
    
            # load image
            file_name = self.dir + '/validation/' + data[1] + '/' + data[0]
            image = Image.open(file_name)
            if(verbose) :
                print("  ", file_name)
                print( '     index  :', self.train_iterator -1)
                print( '     width  :', image.size[0] )
                print( '     height :', image.size[1] )
                print( '     mode   :', image.mode    )
                print( '     format :', image.format  )

            
            # image size test
            if crop and (image.size[0] <= self.size or image.size[1] <= self.size) :
                if(verbose) :
                    print('image too small for cropping (validation) : ', data[1] + '/' + data[0])
            else : 
                break      

        if crop:
            # crop image
            crop_width  = random.randint(0, image.size[0]-self.size-1)
            crop_height = random.randint(0, image.size[1]-self.size-1)
            box = (crop_width, crop_height, crop_width+self.size, crop_height+self.size)
            image = image.crop(box)

            # image transform
            #image.save(self.dir+'/'+str(self.train_iterator -1)+'.jpg')
            orientation = [ Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
            flip = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]

            if (random_flip_flop == True) :
                if (random.choice([True, False])) :
                    image = image.transpose(random.choice(flip))

            if (random_rotate == True) :
                if (random.choice([True, False])) :
                    image = image.transpose(random.choice(orientation))

            # convert the image into a array
            image = np.asarray(image)

            # extract green
            if( self.nb_channels == 1 ) :
                image = self.extract_channel(image,1)
        else: 
            image = np.asarray(image)
        # convert to float image
        image = image.astype(np.float32) / 255.
        #image = image.reshape(1, self.size, self.size, 3)
        image = image.reshape(self.size, self.size, self.nb_channels)

        # buils class label
        label = np.zeros(len(self.image_class))
        pos = self.image_class.index(data[1])
        label[pos] = 1.0
        
        # return image and label
        return (image, label)

        
    def get_batch_validation(self, batch_size = 50, crop = True, random_flip_flop = False, random_rotate = False) :
        
        batch_image = np.empty([batch_size, self.size, self.size, self.nb_channels])
        batch_label = np.empty([batch_size, self.nb_class])
        for i in range(0,batch_size) :
            data = self.get_next_validation(crop, random_flip_flop,random_rotate, verbose=False)
            batch_image[i] = data[0]
            batch_label[i] = data[1]

        return (batch_image.astype(np.float32),batch_label)


    def export_database(self, export_path, nb_train, nb_test, nb_validation, proportion = 0.5):
        print("Exporting training set : " + str(nb_train) + " images to process...")
        batch_size = 100
        i = 0
        n_class0 = 0
        n_class1 = 0

        while(i < nb_train):

            batch = self.get_next_train_batch(batch_size)
            for j in range(batch_size):
                save = True
                exp = Image.fromarray((batch[0][j]*255).astype(np.uint8).reshape(self.size, self.size))
                if batch[1][j][0] == 0.:
                    name_class = self.image_class[0]
                    n_class0 += 1
                    if(n_class0 > int(nb_train/2)):
                        save = False
                else:
                    name_class = self.image_class[1]
                    n_class1 += 1
                    if(n_class1 > int(nb_train/2)):
                        save = False
                if save :
                    exp.save(export_path + '/train/' + name_class + '/' + 'train' + str(i+j) + '.jpg')
                i+=1

            print(str(i) + " images exported")

        print("Exporting testing set : " + str(nb_test) + " images to process...")
        batch_size = 100
        i = 0
        n_class0 = 0
        n_class1 = 0
        while(i < nb_test):
            save = True
            batch = self.get_batch_test(batch_size)
            for j in range(batch_size):
                if batch[1][j][0] == 0.:
                    name_class = self.image_class[0]
                    n_class0 += 1
                    if(n_class0 > int(nb_test/2)):
                        save = False
                else:
                    name_class = self.image_class[1]
                    n_class1 += 1
                    if(n_class1 > int(nb_test/2)):
                        save = False
                if save:
                    exp = Image.fromarray((batch[0][j]*255).astype(np.uint8).reshape(self.size, self.size))
                    exp.save(export_path + '/test/' + name_class + '/' + 'test' + str(i + j) + '.jpg')
                i+=1
            print(str(i) + " images exported")

        print("Exporting validation set : " + str(nb_validation) + " images to process...")
        batch_size = 100
        i = 0
        n_class0 = 0
        n_class1 = 0
        while(i < nb_validation):

            batch = self.get_batch_validation(batch_size)
            for j in range(batch_size):
                if batch[1][j][0] == 0.:
                    name_class = self.image_class[0]
                    n_class0 += 1
                    if(n_class0 > int(nb_validation/2)):
                        save = False
                else:
                    name_class = self.image_class[1]
                    n_class1 += 1
                    if(n_class1 > int(nb_validation/2)):
                        save = False

                if save:
                    exp = Image.fromarray((batch[0][j]*255).astype(np.uint8).reshape(self.size, self.size))
                    exp.save(export_path + '/validation/' + name_class + '/' + 'validation' + str(i + j) + '.jpg')
                i+=1
            print(str(i) + " images exported")

        def get_next_batch(self, category = 'train', batch_size = 50):
            if category == 'train':

                self.train_iterator += 1
            if category == 'test':
                self.test_iterator
            if category == 'validation':
                self.validation_iterator

            
def get_image_filename_from_dir(directory_path) :
    # file extension accepted as image data
    valid_image_extension = [".jpg",".gif",".png",".tga",".tif"]
#    random_prefix = ''.join(random.choice('0123456789ABCDEF') for i in range(7))

    image_list = []

    # load the images
    print("loading images in: ", directory_path)
    for filename in os.listdir(directory_path):

        # check if the file is an image
        extension = os.path.splitext(filename)[1]
        if extension.lower() not in valid_image_extension:
            continue
    
        # open the image 
        image = Image.open(os.path.join(directory_path,filename))
        print( '\n')
        print( '     filename :', filename)
        print( '     width    :', image.size[0] )
        print( '     height   :', image.size[1] )
        print( '     mode     :', image.mode    )
        print( '     format   :', image.format  )
        
        image_list.append(image)

    return image_list


def compute_useless_images(directory_path, image_size, nb_images = 100, treshold = 0.3): 
    data =  Database_loader(directory_path, image_size, only_green=True)  
    
    i = 0
    batch_size = 50
    max_height = np.zeros((nb_images,))
    while(i < nb_images):
        batch = data.get_next_train_batch(batch_size, False)
        ind_batch = 0
        for image in batch[0]:
            image = np.reshape(image, [-1])
            hist = np.histogram(image, 256, [0.,1.])[0]
            # print(hist)
            # print(image)
            # print(batch[1][ind_batch])
            # plt.imshow(np.reshape(image, [image_size, image_size]))
            # plt.show()
            max_height[i] = max(hist)/(image_size**2)
            i+=1
            ind_batch+=1

    nb_useless = 0
    for m in max_height:
        if(m > treshold):
            nb_useless+=1

    print("Number of useless images : " + str(nb_useless) + "/" + str(nb_images))




if __name__ == "__main__":    
    #file_names = load_images('/home/nozick/Desktop/cg_pi_64', 100)
    a = Database_loader('/work/smg/v-nicolas/Test', 200, only_green=True)
#     b = a.get_next_train(random_flip_flop=True,random_rotate=True)
#     print(b[0].shape)
# #    im = Image.fromarray(np.uint8(b[0]*255))
# #    im.save(a.dir+'/caca_'+str(a.train_iterator -1)+'.jpg')

#     c = a.get_next_test()
#     d = a.get_next_validation()
    
    # print('Loading batch')
    # e = a.get_next_train_batch(10)
#    print(e)
    
    a.export_database('/work/smg/v-nicolas/Test_DB_200', nb_train = 0, nb_test = 2000, nb_validation = 1000)

    f = Database_loader('/work/smg/v-nicolas/Test_DB_200', 200, only_green=True)
    # print("Loading batch")
    g = f.get_batch_validation(50, crop = False)

    print(g[0][0].shape)

    # compute_useless_images('/work/smg/v-nicolas/Test_DB_100', 100, nb_images = 1000, treshold = 0.5)
