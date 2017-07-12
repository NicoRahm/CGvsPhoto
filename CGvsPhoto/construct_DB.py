"""
    The ``construct_DB`` module
    ======================
 
    Permits to format an image database to run our code.
"""

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, os.path
import random
import shutil


def load_images_from_dir(dir_name, shuffle = False) :

# file extension accepted as image data
    valid_image_extension = [".jpg", ".jpeg",".gif",".png",".tga",".tif", ".JPG"]

    file_list = []
    nb_image = 0 
    for filename in os.listdir(dir_name):
        # check if the file is an image
        extension = os.path.splitext(filename)[1]
        if extension.lower() in valid_image_extension:
            file_list.append(filename)
            nb_image += 1

    print('    ',nb_image,'images loaded')

    if shuffle: 
    	random.seed(42)
    	random.shuffle(file_list)
    return file_list


def construct_DB(source_real, source_CG, target_dir, nb_per_class = 1800, 
			  	 validation_proportion = 0.1, test_proportion = 0.2): 

	"""Constructs a database with the wright format from CG and Real images
      
  	Selects randomly an equal number of images for each class and divide them into training, 
  	testing and validation step.

  	:param source_real: Directory containing real images
  	:param source_CG: Directory containing CG images
  	:param target_dir: Directory where your database will be saved
  	:param nb_per_class: Total number of images you want for each class
  	:param validation_proportion: Proportion of images in the validation set
  	:param test_proportion: Proportion of images in the testing set

  	:type source_real: str
  	:type source_CG: str
  	:type target_dir: str
  	:type nb_per_class: int
  	:type validation_proportion: float (between 0. and 1.)
  	:type test_proportion: float (between 0. and 1.)
  	"""  

	train_dir = target_dir + 'train/'
	test_dir = target_dir + 'test/'
	validation_dir = target_dir + 'validation/'

	if not os.path.exists(target_dir):
		os.mkdir(target_dir)
		os.mkdir(train_dir)
		os.mkdir(train_dir + 'CGG/')
		os.mkdir(train_dir + 'Real/')
		os.mkdir(test_dir)
		os.mkdir(test_dir + 'CGG/')
		os.mkdir(test_dir + 'Real/')
		os.mkdir(validation_dir)
		os.mkdir(validation_dir + 'CGG/')
		os.mkdir(validation_dir + 'Real/')

	image_real = load_images_from_dir(source_real, shuffle = True)
	image_CG = load_images_from_dir(source_CG, shuffle = True)

	nb_train = int(nb_per_class*(1 - validation_proportion - test_proportion))
	nb_test = int(nb_per_class*test_proportion)
	nb_validation = int(nb_per_class*validation_proportion)

	for i in range(nb_train): 
		shutil.copyfile(source_real + image_real[i], train_dir + 'Real/' + image_real[i])
		shutil.copyfile(source_CG + image_CG[i], train_dir + 'CGG/' + image_real[i])

	print(str(nb_train) + ' training images imported for each class')

	for i in range(nb_train, nb_train + nb_validation): 
		shutil.copyfile(source_real + image_real[i], validation_dir + 'Real/' + image_real[i])
		shutil.copyfile(source_CG + image_CG[i], validation_dir + 'CGG/' + image_real[i])

	print(str(nb_validation) + ' validation images imported for each class')

	for i in range(nb_train + nb_validation, nb_train + nb_validation + nb_test): 
		shutil.copyfile(source_real + image_real[i], test_dir + 'Real/' + image_real[i])
		shutil.copyfile(source_CG + image_CG[i], test_dir + 'CGG/' + image_real[i])	

	print(str(nb_test) + ' testing images imported for each class')

	print("done.")

if __name__ == '__main__': 

	source_real_directory = "/home/nicolas/Database/face_DB/Real/"
	source_CG_directory = "/home/nicolas/Database/face_DB/CGG/"
	target_dir_test = '/home/nicolas/Database/face_DB_split_2/'

	construct_DB(source_real = source_real_directory, 
			  source_CG = source_CG_directory,
			  target_dir = target_dir_test, 
			  nb_per_class = 108,
			  validation_proportion = 0.1, 
			  test_proportion = 0.4)

