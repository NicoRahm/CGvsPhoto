from CGvsPhoto import Model
# to change to your favorite database
database_path = '/work/smg/v-nicolas/level-design_raise_100_color/'
test_data_path = '/work/smg/v-nicolas/face_DB/'

# to change to the format of your image
image_size = 100

# define a single-image classifier
clf = Model(database_path, image_size, config = 'Server', filters = [32,32,64],
            batch_size = 50, feature_extractor = 'Stats', remove_context = True)


# test classifier on total image
clf.test_total_images(test_data_path = test_data_path,
                      nb_images = 216, decision_rule = 'weighted_vote',
                      show_images = False, 
                      save_images = False,
                      only_green = True,
                      other_clf = False)