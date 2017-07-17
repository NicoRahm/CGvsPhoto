"""
    The ``construct_DB`` module
    ======================
 
    Permits to format an image database to run our code.
"""

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, os.path
import random
import shutil
from . import image_loader as il


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


def make_dirs(target_dir):

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

def construct_DB(source_real, source_CG, target_dir, nb_per_class = 1800, 
			  	 validation_proportion = 0.1, test_proportion = 0.2,
			  	 compress = False): 

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

	make_dirs(target_dir)

	train_dir = target_dir + 'train/'
	test_dir = target_dir + 'test/'
	validation_dir = target_dir + 'validation/'

	image_real = load_images_from_dir(source_real, shuffle = True)
	image_CG = load_images_from_dir(source_CG, shuffle = True)

	nb_train = int(nb_per_class*(1 - validation_proportion - test_proportion))
	nb_test = int(nb_per_class*test_proportion)
	nb_validation = int(nb_per_class*validation_proportion)

	for i in range(nb_train): 
		if compress:
			image = Image.open(source_real + image_real[i]) 
			image.save(train_dir + 'Real/' + image_real[i], "JPEG", quality=75)
		else:
			shutil.copyfile(source_real + image_real[i], train_dir + 'Real/' + image_real[i])
		
		shutil.copyfile(source_CG + image_CG[i], train_dir + 'CGG/' + image_real[i])

	print(str(nb_train) + ' training images imported for each class')

	for i in range(nb_train, nb_train + nb_validation): 
		if compress:
			image = Image.open(source_real + image_real[i]) 
			image.save(validation_dir + 'Real/' + image_real[i], "JPEG", quality=75)
		else:
			shutil.copyfile(source_real + image_real[i], validation_dir + 'Real/' + image_real[i])
		
		shutil.copyfile(source_CG + image_CG[i], validation_dir + 'CGG/' + image_real[i])

	print(str(nb_validation) + ' validation images imported for each class')

	for i in range(nb_train + nb_validation, nb_train + nb_validation + nb_test): 
		if compress:
			image = Image.open(source_real + image_real[i]) 
			image.save(test_dir + 'Real/' + image_real[i], "JPEG", quality=75)
		else:		
			shutil.copyfile(source_real + image_real[i], test_dir + 'Real/' + image_real[i])
		
		shutil.copyfile(source_CG + image_CG[i], test_dir + 'CGG/' + image_real[i])	

	print(str(nb_test) + ' testing images imported for each class')

	print("done.")


def construct_Kfold(source_DB, target_dir, K = 5, size_patch = 100):


	print('Construction K-Fold...\n')
	os.mkdir(target_dir)
	source_real = source_DB + '/Real/'
	source_CG = source_DB + '/CGG/'
	image_real = load_images_from_dir(source_real, shuffle = True)
	image_CG = load_images_from_dir(source_CG, shuffle = True)

	nb_image_real = len(image_real)
	nb_image_CG = len(image_CG)

	nb_test_real = int(nb_image_real/K)
	nb_test_CG = int(nb_image_CG/K)

	nb_validation_real = int(nb_image_real/10)
	nb_validation_CG = int(nb_image_CG/10)

	nb_train_real = nb_image_real - nb_validation_real - nb_test_real
	nb_train_CG = nb_image_CG - nb_validation_CG - nb_test_CG

	for k in range(K): 

		name_DB = 'Fold_' + str(k+1) + '/'
		fold_dir = target_dir + '/' + name_DB
		make_dirs(fold_dir)
		train_dir = fold_dir + 'train/'
		test_dir = fold_dir + 'test/'
		validation_dir = fold_dir + 'validation/'

		print('Exporting Fold ' + str(k+1))
		
		for i in range(k*nb_test_real, (k+1)*nb_test_real):
			shutil.copyfile(source_real + image_real[i%nb_image_real], test_dir + 'Real/' + image_real[i%nb_image_real])
		for i in range(k*nb_test_CG, (k+1)*nb_test_CG):
			shutil.copyfile(source_CG + image_CG[i%nb_image_CG], test_dir + 'CGG/' + image_CG[i%nb_image_CG])

		for i in range((k+1)*nb_test_real, (k+1)*nb_test_real + nb_validation_real):
			shutil.copyfile(source_real + image_real[i%nb_image_real], validation_dir + 'Real/' + image_real[i%nb_image_real])
		for i in range((k+1)*nb_test_CG, (k+1)*nb_test_CG + nb_validation_CG):
			shutil.copyfile(source_CG + image_CG[i%nb_image_CG], validation_dir + 'CGG/' + image_CG[i%nb_image_CG])

		for i in range((k+1)*nb_test_real + nb_validation_real, (k+1)*nb_test_real + nb_validation_real + nb_train_real):
			shutil.copyfile(source_real + image_real[i%nb_image_real], train_dir + 'Real/' + image_real[i%nb_image_real])
		for i in range((k+1)*nb_test_CG + nb_validation_CG, (k+1)*nb_test_CG + nb_validation_CG + nb_train_CG):
			shutil.copyfile(source_CG + image_CG[i%nb_image_CG], train_dir + 'CGG/' + image_CG[i%nb_image_CG])

		print('Exporting patch database...')
		DB = il.Database_loader(fold_dir, size_patch, 
                        	 	only_green=False, rand_crop = True)
    
		DB.export_database(target_dir + '/Patch_DB_' + str(k+1) + '/', 
						   nb_train = 80000, 
						   nb_test = 4000, 
						   nb_validation = 2000)


if __name__ == '__main__': 

	source_real_directory = "/home/nicolas/Database/Source_level-design_raise/Real/"
	source_CG_directory = "/home/nicolas/Database/Source_level-design_raise/CGG/"
	target_dir_test = '/home/nicolas/Database/level-design_raise_compress/'

	construct_DB(source_real = source_real_directory, 
			  source_CG = source_CG_directory,
			  target_dir = target_dir_test, 
			  nb_per_class = 1800,
			  validation_proportion = 0.1, 
			  test_proportion = 0.2,
			  compress = True)

