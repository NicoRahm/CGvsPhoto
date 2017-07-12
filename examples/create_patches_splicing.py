from CGvsPhoto import Database_loader

# directory with the original database
source_db = '/home/nicolas/Database/face_DB_split_2/'

# wanted size for the patches 
image_size = 100

# directory to store the patch database
target_patches = '/home/nicolas/Database/face_DB_100_2/'


# create a database manager 
DB = Database_loader(source_db, image_size, 
                     only_green=False, rand_crop = True)

# export a patch database    
DB.export_database(target_patches, 
                   nb_train = 10000, 
                   nb_test = 8000, 
                   nb_validation = 1000)

# directory to store splicing images 
# target_splicing = '/home/nicolas/Database/splicing2/'


# DB.export_splicing(target_splicing, 50)
