from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, os.path
import random
import shutil


def load_images_from_dir(dir_name, shuffle = False) :

# file extension accepted as image data
    valid_image_extension = [".jpg",".gif",".png",".tga",".tif", ".JPG"]

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


def construct_DB(source_real, source_CG, target_dir, nb_total = 1800, 
			  validation_proportion = 0.1, test_proportion = 0.2): 



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

	nb_train = int(nb_total*(1 - validation_proportion - test_proportion))
	nb_test = int(nb_total*test_proportion)
	nb_validation = int(nb_total*validation_proportion)

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

	source_real_directory = "/home/nicolas/Database/dresden/jpeg/"
	source_CG_directory = "/home/nicolas/Database/GameCG/"
	target_dir_test = '/home/nicolas/Database/level-design_dresden/'

	construct_DB(source_real = source_real_directory, 
			  source_CG = source_CG_directory,
			  target_dir = target_dir_test, 
			  nb_total = 1800,
			  validation_proportion = 0.1, 
			  test_proportion = 0.2)

