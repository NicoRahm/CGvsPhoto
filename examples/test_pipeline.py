import sys
sys.path.append('../')

from model import Model

database_path = '/home/nicolas/Database/level-design_raise_100/'

image_size = 100

clf = Model(database_path, image_size, config = 'Personal', filters = [4,8],
            batch_size = 50, feature_extractor = 'Stats')

clf.train(nb_train_batch = 100,
          nb_test_batch = 80, 
          nb_validation_batch = 40,
          show_filters = False)

