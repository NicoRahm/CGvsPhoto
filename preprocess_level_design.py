
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, os.path



def load_images_in_dir(dir_name) :

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

    return file_list

def process_images(file_list, data_dir, target_dir, box = (None, None, None, None)): 


	nb_images = 0
	for file_name in file_list:

		image = Image.open(data_dir + file_name)
		(width, height) = image.size

		if box[0] == None: 
			b0 = 0
		else: 
			b0 = box[0]

		if box[1] == None: 
			b1 = 0
		else:
			b1 = box[1]

		if box[2] == None: 
			b2 = width
		else: 
			b2 = width - box[2]

		if box[3] == None:
			b3 = height
		else:
			b3 = height - box[3] 

		image = image.crop((b0,b1,b2,b3))
		image.save(target_dir + file_name)
		nb_images += 1

	print('     %i images processed' %nb_images)



if __name__ == '__main__':

	data_dir = '/home/nicolas/Documents/data_level_design/'
	target_dir = '/home/nicolas/Documents/GameCG/'

	# Name of the directory to process
	game = 'Witcher 3'
	data_dir += game + '/'

	file_list = load_images_in_dir(data_dir)

	process_images(file_list, data_dir, target_dir, box = (None, None, 520, 250))


