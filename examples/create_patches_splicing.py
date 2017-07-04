from CGvsPhoto import Database_loader

# directory with the original database
source_db = '/home/nicolas/Database/level-design_raise/'

# wanted size for the patches 
image_size = 100

# directory to store the patch database
target_patches = '/home/nicolas/Database/level-design_raise_100_color/'


# create a database manager 
DB = Database_loader(source_db, image_size, 
                     only_green=False, rand_crop = True)

# export a patch database    
DB.export_database(target_patches, 
                   nb_train = 80000, 
                   nb_test = 4000, 
                   nb_validation = 2000)

# directory to store splicing images 
# target_splicing = '/home/nicolas/Database/splicing2/'


# DB.export_splicing(target_splicing, 50)
