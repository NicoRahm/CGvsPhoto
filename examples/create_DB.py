from CGvsPhoto import construct_DB

# Change to the source of real images
source_real = "/home/nicolas/Database/dresden/jpeg/"
# Change to the source of CG images
source_CG = "/home/nicolas/Database/GameCG/"
# The directory where the database will be saved
target_dir = '/home/nicolas/Database/test_DB/'

# Construct a database with an equilibrated CG/Real ratio
# Formatted to be used with the image_loader
construct_DB(source_real = source_real, 
			  source_CG = source_CG,
			  target_dir = target_dir, 
			  nb_per_class = 100,
			  validation_proportion = 0.1, 
			  test_proportion = 0.2)
