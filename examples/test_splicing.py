from CGvsPhoto import Model
# to change to your splicing database
splicing_data_path = '/home/nicolas/Database/faces_celeb/'

# to change to your training database
database_path = '/home/nicolas/Database/level-design_raise/'

# to change to the format of your patches
image_size = 100

# define a single-image classifier
clf = Model(database_path, image_size, config = 'Personal', filters = [32,64],
            batch_size = 50, feature_extractor = 'Stats')


# tests classifier on splicing images
# you have to load pre-trained weights 
clf.test_splicing(data_path = splicing_data_path, 
                  nb_images = 50,
                  minibatch_size = 25,
                  show_images = True,
                  save_images = False)