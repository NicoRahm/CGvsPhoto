from CGvsPhoto.construct_DB import construct_Kfold

source_db = '/home/nicolas/Database/Source_level-design_raise/'
target_dir = '/home/nicolas/Database/Kfold_level-design_raise/'

construct_Kfold(source_db, target_dir, K = 5, size_patch = 100)
