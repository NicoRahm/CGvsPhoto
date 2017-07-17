from CGvsPhoto import Model
import os 
# to change to your favorite database
database_path = '/work/smg/v-nicolas/level-design_raise_100_color/'
test_data_path = '/work/smg/v-nicolas/level-design_raise_compress/test/'
# test_data_path = '/work/smg/v-nicolas/face_DB_split/test/'

# to change to the format of your image
image_size = 100

# define a single-image classifier
clf = Model(database_path, image_size, config = 'Server', filters = [32,32,64],
            batch_size = 50, feature_extractor = 'Stats', remove_context = True,
            remove_filter_size = 5)


# nb_images = len([name for name in os.listdir(test_data_path + '/CGG') if os.path.isfile(name)]) + len([name for name in os.listdir(test_data_path + '/Real') if os.path.isfile(name)])
# test classifier on total image
clf.test_total_images(test_data_path = test_data_path,
                      nb_images = 720, 
                      decision_rule = 'weighted_vote',
                      show_images = False, 
                      save_images = False,
                      only_green = True,
                      other_clf = False)