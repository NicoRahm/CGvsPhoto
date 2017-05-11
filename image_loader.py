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

    def __init__(self, directory, size, proportion = 1.0, seed=42, only_green=True, rand_crop = True) :

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
        self.image_class = ['Real', 'CGG']         # list of the class (label) used in the process
        self.nb_class = 0
        self.train_iterator = 0       # iterator over the train images
        self.test_iterator = 0        # iterator over the test images
        self.validation_iterator = 0  # iterator over the validation images
        self.rand_crop = rand_crop
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
        # self.image_class = self.get_immediate_subdirectories(train_dir_name)
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


    def get_next_train(self, crop = True, rand_crop = True, random_flip_flop = False, random_rotate = False, verbose = False) :

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
            if rand_crop: 
                crop_width  = random.randint(0, image.size[0]-self.size-1)
                crop_height = random.randint(0, image.size[1]-self.size-1)
            else: 
                crop_width = 0
                crop_height = 0

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

            if( self.nb_channels == 1 and len(image.shape) > 2 ) :
                image = self.extract_channel(image,1)
        # convert to float image
        image = image.astype(np.float32) / 255.
        #image = image.reshape(1, self.size, self.size, 3)
        if self.size == None: 
            image = image.reshape(image.shape[0], image.shape[1], self.nb_channels)
        else: 
            image = image.reshape(self.size, self.size, self.nb_channels)

        # build class label
        label = np.zeros(len(self.image_class))
        pos = self.image_class.index(data[1])
        label[pos] = 1.0
        
        # return image and label
        return (image, label)

        
    def get_next_train_batch(self, batch_size = 50, crop = True, random_flip_flop = False, random_rotate = False) :

        batch_image = np.empty([batch_size, self.size, self.size, self.nb_channels])
        batch_label = np.empty([batch_size, self.nb_class])
        for i in range(0,batch_size) :
            data = self.get_next_train(crop, self.rand_crop, random_flip_flop,random_rotate, verbose=False)
            batch_image[i] = data[0]
            batch_label[i] = data[1]

        return (batch_image.astype(np.float32),batch_label)

        
    def get_next_test(self, crop = True, rand_crop = True, random_flip_flop = False, random_rotate = False, verbose = False) :

        
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
            if rand_crop: 
                crop_width  = random.randint(0, image.size[0]-self.size-1)
                crop_height = random.randint(0, image.size[1]-self.size-1)
            else: 
                crop_width = 0 
                crop_height = 0

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

            if( self.nb_channels == 1 and len(image.shape) > 2 ) :
                image = self.extract_channel(image,1)
        # convert to float image
        image = image.astype(np.float32) / 255.
        #image = image.reshape(1, self.size, self.size, 3)
        if self.size == None: 
            image = image.reshape(image.shape[0], image.shape[1], self.nb_channels)
        else: 
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
            data = self.get_next_test(crop, self.rand_crop, random_flip_flop,random_rotate)
            batch_image[i] = data[0]
            batch_label[i] = data[1]

        return (batch_image.astype(np.float32),batch_label)
      

    def get_next_validation(self, crop = True, rand_crop = True, random_flip_flop = False, random_rotate = False, verbose = False) :

        
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
            if rand_crop: 
                crop_width  = random.randint(0, image.size[0]-self.size-1)
                crop_height = random.randint(0, image.size[1]-self.size-1)
            else: 
                crop_width = 0
                crop_height = 0

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
            data = self.get_next_validation(crop, self.rand_crop, random_flip_flop,random_rotate, verbose=False)
            batch_image[i] = data[0]
            batch_label[i] = data[1]

        return (batch_image.astype(np.float32),batch_label)

    def export_splicing(self, export_path, nb_images, radius = 300): 

        batch_size = 10
        if not os.path.exists(export_path):
            os.mkdir(export_path)
        i = 0
        while(i < nb_images):
            batch = []
            batch.append([])
            batch.append([])
            k1 = 0
            k2 = 0
            while k1 < batch_size or k2 < batch_size:

                data = self.file_test[self.test_iterator]
                self.test_iterator += 1
                if self.test_iterator >= len(self.file_test) :
                    self.test_iterator = 0

                file_name = self.dir + '/test/' + data[1] + '/' + data[0]
                image = self.extract_channel(np.array(Image.open(file_name)), 1)
                # print(data[1])
                if data[1] == 'Real' and k1 < batch_size:
                    batch[0].append(image)
                    k1 += 1
                if data[1] == 'CGG' and k2 < batch_size:
                    batch[1].append(image)
                    k2 += 1

            for j in range(batch_size):
                
                image_real = batch[0][j]
                image_cgg = batch[1][j]

                shape_cgg = image_cgg.shape
                shape_real = image_real.shape
                adding = np.zeros(shape_real)
                r = radius
                a_cgg, b_cgg = random.randint(radius, shape_cgg[0] - radius), random.randint(radius, shape_cgg[1] - radius)
                
                y,x = np.ogrid[-a_cgg:shape_cgg[0]-a_cgg, -b_cgg:shape_cgg[1]-b_cgg]
                mask_cgg = x*x + y*y <= r*r

                a_real, b_real = random.randint(radius, shape_real[0] - radius), random.randint(radius, shape_real[1] - radius)                


                y,x = np.ogrid[-a_real:shape_real[0]-a_real, -b_real:shape_real[1]-b_real]
                mask_real = x*x + y*y <= r*r
                image_real[mask_real] = 0

                adding[mask_real] = image_cgg[mask_cgg]

                result = image_real + adding

                print(result.shape)

                exp = Image.fromarray(result).convert('RGB')
                exp.save(export_path + str(i) + '.jpg')
                i+=1

            print(str(i) + " images exported")


    def export_database(self, export_path, nb_train, nb_test, nb_validation, proportion = 0.5):

        train_dir = export_path + 'train/'
        test_dir = export_path + 'test/'
        validation_dir = export_path + 'validation/'
        if not os.path.exists(export_path):
            os.mkdir(export_path)

            os.mkdir(train_dir)
            os.mkdir(train_dir + 'CGG/')
            os.mkdir(train_dir + 'Real/')

            os.mkdir(test_dir)
            os.mkdir(test_dir + 'CGG/')
            os.mkdir(test_dir + 'Real/')

            os.mkdir(validation_dir)
            os.mkdir(validation_dir + 'CGG/')
            os.mkdir(validation_dir + 'Real/')

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
                    name_class = self.image_class[1]
                    n_class0 += 1
                    if(n_class0 > int(nb_train/2)):
                        save = False
                else:
                    name_class = self.image_class[0]
                    n_class1 += 1
                    if(n_class1 > int(nb_train/2)):
                        save = False
                if save :
                    exp.save(export_path + '/train/' + name_class + '/' + 'train' + str(i) + '.jpg')
                    i+=1

            print(str(i) + " images exported")

        print("Exporting testing set : " + str(nb_test) + " images to process...")
        batch_size = 100
        i = 0
        n_class0 = 0
        n_class1 = 0
        while(i < nb_test):
            
            batch = self.get_batch_test(batch_size)
            for j in range(batch_size):
                save = True
                if batch[1][j][0] == 0.:
                    name_class = self.image_class[1]
                    n_class0 += 1
                    if(n_class0 > int(nb_test/2)):
                        save = False
                else:
                    name_class = self.image_class[0]
                    n_class1 += 1
                    if(n_class1 > int(nb_test/2)):
                        save = False
                if save:
                    exp = Image.fromarray((batch[0][j]*255).astype(np.uint8).reshape(self.size, self.size))
                    exp.save(export_path + '/test/' + name_class + '/' + 'test' + str(i) + '.jpg')
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
                save = True
                if batch[1][j][0] == 0.:
                    name_class = self.image_class[1]
                    n_class0 += 1
                    if(n_class0 > int(nb_validation/2)):
                        save = False
                else:
                    name_class = self.image_class[0]
                    n_class1 += 1
                    if(n_class1 > int(nb_validation/2)):
                        save = False

                if save:
                    exp = Image.fromarray((batch[0][j]*255).astype(np.uint8).reshape(self.size, self.size))
                    exp.save(export_path + '/validation/' + name_class + '/' + 'validation' + str(i) + '.jpg')
                    i+=1
            print(str(i) + " images exported")


class Test_loader: 

    def __init__(self, directory, subimage_size, only_green = True):

        self.dir = directory          # directory with the train / test / validation sudirectories
        self.subimage_size = subimage_size              # size of the sub image that should be croped
        self.nb_channels = 3          # return only the green channel of the images
        if(only_green == True) :
            self.nb_channels = 1
        self.iterator = 0        # iterator over the test images
        self.validation_iterator = 0  # iterator over the validation images
        self.seed = 42

        self.image_class = ['Real', 'CGG']
        self.nb_class = len(self.image_class)
        print('     number of classes :', self.nb_class, '   ', self.image_class)

        self.file_test = self.load_images_in_dir(self.dir,self.image_class)
        

    def get_immediate_subdirectories(self,a_dir) :
        # return the list of sub directories of a directory
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]


    def load_images_in_dir(self, dir_name, image_class) :

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

            for i in range(int(len(file_list_by_class))):
                file_list.append((file_list_by_class[i],c))
                nb_image_per_class += 1
            print('    ',c,nb_image_per_class,'images loaded')
        random.seed(self.seed)
        random.shuffle(file_list)
        return file_list


    def extract_subimages(self, image_file, subimage_size):

        image = Image.open(self.dir + image_file)
        subimages = []
        width = image.size[0]
        height = image.size[1]

        current_height = 0
        
        while current_height + subimage_size <= height: 
            current_width = 0
            while current_width + subimage_size <= width: 
                box = (current_width, current_height, 
                       current_width + subimage_size, 
                       current_height + subimage_size)
                sub = np.asarray(image.crop(box))
                if len(sub.shape) > 2: 
                    subimages.append(sub[:,:,1].astype(np.float32)/255)
                else: 
                    subimages.append(sub.astype(np.float32)/255)
                current_width += subimage_size

            current_height += subimage_size

        nb_subimages = len(subimages)
        print('Image of size ' + str(width) + 'x' + str(height) + 
              ' cropped at ' + str(subimage_size) + 'x' + str (subimage_size) + 
              ' : ' + str(nb_subimages) + ' outputed subimages.')
        return((np.reshape(np.array(subimages), (nb_subimages, subimage_size, subimage_size, 1)), width, height))

    def get_next_image(self):

        if self.iterator >= len(self.file_test):
            self.iterator = 0
        labeled_image = self.file_test[self.iterator]
        image_file = labeled_image[1] + '/' + labeled_image[0]
        self.iterator += 1

        subimages, width, height = self.extract_subimages(image_file, self.subimage_size)

        return((subimages, labeled_image[1], width, height))
          
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

    source_db = '/home/nicolas/Database/level-design_raise/'
    image_size = 100
    target_db = '/home/nicolas/Database/level-design_raise_100/'

    a = Database_loader(source_db, image_size, only_green=True, rand_crop = False)
    
    # a.export_database(target_db, 
    #                   nb_train = 40000, 
    #                   nb_test = 4000, 
    #                   nb_validation = 2000)
    target_splicing = '/home/nicolas/Database/splicing/'
    a.export_splicing(target_splicing, 50)
    # f = Database_loader(target_db, image_size, only_green=True)

    # g = f.get_batch_validation(50, crop = False)

    # print(g[0][0].shape)

    # compute_useless_images('/work/smg/v-nicolas/Test_DB_100', 100, nb_images = 1000, treshold = 0.5)
    
    source_test = '/home/nicolas/Database/level-design_raise/test/'
    test = Test_loader(source_test, subimage_size = 100)

    test.get_next_image()