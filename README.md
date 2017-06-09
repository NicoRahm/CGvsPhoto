# Computer Graphics vs Real Photographic Images : A Deep-learning approach

**CGvsPhoto** implements a method for computer graphics detection using Convolutional Neural Networks with TensorFlow back-end. The package contains methods for extracting patches from computer graphics and real images, training a CNN with a custom statistical layer, testing this model, comparing with a [state of the art method](http://ieeexplore.ieee.org/abstract/document/6115849/), visualizing probability maps, etc.

![splicing](https://user-images.githubusercontent.com/17125992/26917538-9d918318-4c69-11e7-8c6f-f865b3c5f063.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes. 

### Prerequisites
* Python 3.6+
* Numpy 1.6.1+
* Scikit-learn 0.18.1+ 
* TensorFlow 1.0.1+ (https://github.com/tensorflow/tensorflow)
* Pillow 3.1.2+
* Matplotlib 1.3.1+


### Installing

Simply clone the repository into your favorite directory.

```
$ git clone https://github.com/NicoRahm/CGvsPhoto
```
Then, move to the directory and install the package using : 
```
$ cd CGvsPhoto/
$ pip3 install .
```

To run the first test, there is one more thing to set up : 
* Create a file named config.ini in your execution directory (the directory containing the scripts) to store the different configurations of your environment. The format is the following :
```
[Name of the configuration]
dir_ckpt = /path/to/save/trained/weights/
dir_summaries = /path/to/save/summaries
dir_visualization = /path/to/save visualizations
```

### Database format 

Your database must follow this organization : 
```
Database/
    test/
        CGG/
        Real/
    train/
        CGG/
        Real/
    validation/ 
        CGG/
        Real/
```
You can create it manually or using the function construct_DB. 

### Some simple examples

To get started, you can run simple scripts from the examples directory. Do not forget to **set up the config.ini file** correctly as described above and to modify the pathes to data.

* [create_DB.py](examples/create_DB.py) will create a formated database for future tests.
* [create_patches_splicing.py](examples/create_patches_splicing.py) will create a patches database for training single-image classifier and a splicing database to test our models.
* [test_pipeline.py](examples/test_pipeline.py) trains a neural network to classify image patches and then evaluate it.
* [test_splicing.py](examples/test_splicing.py) tests a model on spliced images.

## How to use

This section explains basic uses of this code. We will describe a step by step procedure to evaluate our model on your database. 

### Formatting the database 

As our code uses a special format for the database, the first thing you need to do is to create a suited structure for the data. You can do this manually but we give a piece of code to do it automatically which may prevent bad surprises... It will create validation, training and testing directories and put a certain number of images per class in it (same number of image for each class)
To do so, you just need to have CG and PG images in two different directories and choose a directory to store the formatted database. Then you can just use the *construct_DB* method : 

```python
from CGvsPhoto import construct_DB

path_CG = '/path/to/CG'
path_PG = '/path/to/PG'
path_export = 'path/to/export/database'

construct_DB(source_real = path_PG, source_CG = path_CG,
             target_dir = path_export, nb_per_class = 1000,
             validation_proportion = 0.1, test_proportion = 0.2)
```

You can choose the total number of images per class with the nb_per_class parameter and the proportion of images to put in each directory.

### Creating the patches database 

Our implementation computes local patch classification before aggregating the results for full-size images. So, to train the single-image classifier, a patch database must be created. 
Our solution is to use the *Database_loader* class : 

```python
from CGvsPhoto import Database_loader

path_source = 'path/to/source/database'
path_export = 'path/to/export/patches'
size_patch = 100

DB = Database_loader(path_source, image_size = size_patch, 
                     only_green=True)

# export a patch database    
DB.export_database(path_export, 
                   nb_train = 40000, 
                   nb_test = 4000, 
                   nb_validation = 2000)
```

Here, you can choose the size of your patches (100x100 pixels in our initial implementation) and the number of patches for each directory (with 50/50 distribution between each class). 

Note that accepted extensions are [".jpg",".gif",".png",".tga",".tif", ".JPG", ".jpeg"]

### Creating a model

Now comes the fun part : to create your own model, you just have to call the *Model* class. For example : 

```python
from CGvsPhoto import Model

model = Model(database_path 'Database/My_Patch_Data', image_size = 100,
              config = 'Config1', filters = [32, 64], 
              feature_extractor = 'Stats', batch_size = 50)
```

You can specify the number of output filtered images for each layer with the parameter filter and the feature extraction scheme (between 'Hist' and 'Stats'). 

Warning : The database must contain images with the same image_size as specified in parameter image_size. 

### Training a classifier

Now, to train this model, use the *train* function specifying the number of training/validation/testing batches: 

```python
model.train(nb_train_batch = 15000,
            nb_test_batch = 80, 
            nb_validation_batch = 40)
```

This will train a model and save the weights and a bunch of summaries in correspondant directories (you specify the name of the run at the begining of the procedure). You can also load a pre-trained model and continue the training (be careful though to load a model which structure corresponds to the one you are trying to train).

At the end of training, the model's accuracy is evaluated on the patches testing set.

### Testing

Now that you have trained a model, you can load it and test it on full-size images, using the *test_total_images* function :

```python
test_data_path = '/Database/My_Data/test/'
clf.test_total_images(test_data_path = test_data_path,
                      nb_images = 720, decision_rule = 'weighted_vote')
```
Your test directory must contain two sub-directories : CGG and Real. 
Before testing, the console will ask you the name of the weight file to load. It must be in the default checkpoint directory and you should inidcate the .ckpt file.
You can specify the number of images you want to process and the aggregation scheme between 'weighted_vote' and 'majority_vote' (even if 'weighted_vote' is in general more efficient).

## Authors

**Nicolas Rahmouni**  - [NicoRahm](https://github.com/NicoRahm)

**Vincent Nozick**

