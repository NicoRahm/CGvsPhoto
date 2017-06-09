import os, os.path
import random

    
def file_shuffler_from_dir(directory_path, percentage_test, percentage_validation) : 
    
    # check for existing output directory for test and validation
    validation_dir_name = directory_path + '/validation'
    if os.path.exists(validation_dir_name):
        print("error: validation directory already exists")
        return

    test_dir_name = directory_path + '/test'
    if os.path.exists(test_dir_name):
        print("error: test directory already exists")
        return
        
    # create an output directory for test and validation
    os.makedirs(validation_dir_name)
    os.makedirs(test_dir_name)   
   
    # count files in 'directory_path'
    num_files = sum(os.path.isfile(os.path.join(directory_path, f)) for f in os.listdir(directory_path))
    print('number of files :', num_files )
    
    # get the files name in the directory
    file_list = []
    for filename in os.listdir(directory_path):
        file_list.append(filename)

    # shuffle the list
    random.shuffle(file_list)
    
    # move the files
    nb_files_validation = int(num_files * percentage_validation)
    print('files to validation :', nb_files_validation )
    for i in range(nb_files_validation) :
        filename = file_list[i]
        path_source = os.path.join(directory_path, filename)
        path_dest = os.path.join(validation_dir_name, filename)
        os.rename(path_source, path_dest)

    nb_files_test = int(num_files * percentage_test)
    print('files to test :', nb_files_test )    
    for i in range(nb_files_test) :
        filename = file_list[-i-1]
        path_source = os.path.join(directory_path, filename)
        path_dest = os.path.join(test_dir_name, filename)
        os.rename(path_source, path_dest)
    
    return
    

    
    
def file_shuffler_link(input_dir, ouput_dir, class_name='', percentage_test=0.1, percentage_validation=0.1, max_size=-1) :
      
    print('create missing directories ...')
    # create the "train / valid / test" directory architecture and clas directory
    if( os.path.exists( ouput_dir + '/train') == False ):
        os.makedirs( ouput_dir + '/train' )

    if( os.path.exists( ouput_dir + '/validation') == False ):
        os.makedirs( ouput_dir + '/validation' )

    if( os.path.exists( ouput_dir + '/test') == False ):
        os.makedirs( ouput_dir + '/test' )
        
    # create class directory   
    if(class_name == '') :
        class_name = os.path.basename(input_dir)

    if( os.path.exists( ouput_dir + '/train/' + os.path.basename(input_dir)) == False ):
        os.makedirs( ouput_dir + '/train/' + class_name )
        
    if( os.path.exists( ouput_dir + '/validation/' + os.path.basename(input_dir)) == False ):
        os.makedirs( ouput_dir + '/validation/' + class_name )
        
    if( os.path.exists( ouput_dir + '/test/' + os.path.basename(input_dir)) == False ):
        os.makedirs( ouput_dir + '/test/' + class_name )        
        
        
    # count files in 'directory_path'
    print('\ncount input files ... ')
    print('  directory :', input_dir)
    num_files = sum(os.path.isfile(os.path.join(input_dir, f)) for f in os.listdir(input_dir))
    print('  number of files :', num_files )
    
    # file extension accepted as image data
    valid_image_extension = [".jpg",".gif",".png",".tga",".tif"]

    # get the files name in the directory
    print('\nget the list of image file ...')
    file_list = []
    for filename in os.listdir(input_dir):
        extension = os.path.splitext(filename)[1]
        if extension.lower() in valid_image_extension:
            file_list.append(filename)  
    
    num_images = len(file_list)
    print('  images found :', num_images)
    
       
    # shuffle the list
    print('\nshuffle the list ...')
    random.shuffle(file_list)       
    
    # max sie
    if(max_size == -1) :
        max_size = num_images
    else :
        num_images = min(num_images,max_size)
    print('  images used :', num_images)
    
    # create the links
    print('\ncreate the links ...')

    # validation
    nb_files_validation = int(num_images * percentage_validation)
    index = 0
    print('  files to validation :', nb_files_validation )
    for i in range(nb_files_validation) :
        filename = file_list[index + i]
        path_source = os.path.join(input_dir, filename)
        random_prefix = ''.join(random.choice('0123456789ABCDEF') for i in range(5))
        path_dest = ouput_dir + '/validation/' + class_name + '/' + random_prefix + '_' + filename
        os.symlink(path_source, path_dest)
    index += nb_files_validation

    # test
    nb_files_test = int(num_images * percentage_test)
    print('  files to test :', nb_files_test )
    for i in range(nb_files_test) :
        filename = file_list[index + i]
        path_source = os.path.join(input_dir, filename)
        random_prefix = ''.join(random.choice('0123456789ABCDEF') for i in range(5))
        path_dest = ouput_dir + '/test/' + class_name + '/' + random_prefix + '_' + filename
        os.symlink(path_source, path_dest)
    index += nb_files_test

    # train
    nb_files_train = num_images - nb_files_validation - nb_files_test
    print('  files to train :', nb_files_train )
    for i in range(nb_files_train) :
        filename = file_list[index + i]
        path_source = os.path.join(input_dir, filename)
        random_prefix = ''.join(random.choice('0123456789ABCDEF') for i in range(5))
        path_dest = ouput_dir + '/train/' + class_name + '/' + random_prefix + '_' + filename
        os.symlink(path_source, path_dest)
    index += nb_files_train
        
    
    print('\ndone.\n\n')

    
    
    

if __name__ == "__main__":    

#    file_shuffler_link('/home/nozick/Desktop/database/images/original/Games/images',
#                       '/home/nozick/Desktop/database/cg_pi_64/test7',
#                       class_name='youtube_cgg')
#
#    file_shuffler_link('/home/nozick/Desktop/database/images/original/Movie/images',
#                       '/home/nozick/Desktop/database/cg_pi_64/test7',
#                       class_name='youtube_pi')
#   
#    file_shuffler_link('/home/nozick/Desktop/database/images/original/gopro/images',
#                       '/home/nozick/Desktop/database/cg_pi_64/test7',
#                       class_name='youtube_pi2')
    
    
   file_shuffler_link('/home/nozick/Desktop/database/images/original/dario_database/cgg',
                      '/home/nozick/Desktop/database/cg_pi_64/test8',
                      class_name='dario_cgg',
                      max_size=2600) 
    
   file_shuffler_link('/home/nozick/Desktop/database/images/original/Dresden_Dataset/jpeg',
                      '/home/nozick/Desktop/database/cg_pi_64/test8',
                      class_name='dresden',
                      max_size=2600) 
    
    
    
    
    
    
    
    
    
    
    
    