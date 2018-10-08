Computer Graphics vs Real Photographic Images : A Deep-learning approach
========================================================================

.. image:: https://badge.fury.io/py/CGvsPhoto.svg
    :target: https://badge.fury.io/py/CGvsPhoto
    
**CGvsPhoto** implements a method for computer graphics detection using
Convolutional Neural Networks with TensorFlow back-end. The package
contains methods for extracting patches from computer graphics and real
images, training a CNN with a custom statistical layer, testing this
model, comparing with a `state of the art method`_, visualizing
probability maps, etc. See the paper corresponding to this model `here`_.

.. figure:: https://user-images.githubusercontent.com/17125992/26917538-9d918318-4c69-11e7-8c6f-f865b3c5f063.png
   :alt: splicing

Dataset
---------------

The dataset used for testing our method is composed of 1800 real photographs and 1800 Video-game images. The photographs were randomly taken from the `Raise Database`_ and converted to JPEG format (compression rate 95%). The names of the images used during our experiments for each set (training, testing and validation) are available on the corresponding .csv files in the `data directory`_.  

The Video-game images were downloaded from the `Level-design reference Database`_ and cropped to remove in-game informations. They were extracted from 5 of the most photorealistic video-games:  The Witcher 3, Battlefield 4, Battlefield Bad Company  2, Grand Theft Auto 5 and Uncharted 4. To download the preprocessed images, follow this `link`_. 

Getting Started
---------------

These instructions will get you a copy of the project up and running on
your local machine for testing purposes.

Prerequisites
~~~~~~~~~~~~~

-  Python 3.6+
-  Numpy 1.6.1+
-  Scikit-learn 0.18.1+
-  TensorFlow 1.0.1+ (https://github.com/tensorflow/tensorflow)
-  Pillow 3.1.2+
-  Matplotlib 1.3.1+

Installing
~~~~~~~~~~

Simply install this package with pip3 : 
::

    $ pip3 install CGvsPhoto

You can also clone the repository into your favorite directory.

::

    $ git clone https://github.com/NicoRahm/CGvsPhoto

Then, install the package using :

::

    $ cd CGvsPhoto/
    $ pip3 install .

To run your first test, there is one more thing to set up:

- Create a file named config.ini in your execution directory (the directory containing your scripts) to store the different configurations of your environment. The format is the following :

::

    [Name of the configuration]
    dir_ckpt = /path/to/save/trained/weights/
    dir_summaries = /path/to/save/summaries
    dir_visualization = /path/to/save visualizations

An example file is given in the examples directory.

Database format
~~~~~~~~~~~~~~~

Your database must follow this organization :

::

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

You can create it manually or use the function construct\_DB.

Some simple examples
~~~~~~~~~~~~~~~~~~~~

To get started, you can run simple scripts from the examples directory.
Do not forget to **set up the config.ini file** correctly as described
above and to modify the paths to data.

-  `create\_DB.py`_ will create a formated database for future tests.
-  `create\_patches\_splicing.py`_ will create a patches database for
   training single-image classifier and a splicing database to test our
   models.
-  `test\_pipeline.py`_ trains a neural network to classify image
   patches and then evaluate it.
-  `test\_splicing.py`_ tests a model on spliced images.

How to use
----------

This section explains basic uses of this code. We describe a step by
step procedure to evaluate our model on your database.

Formatting the database
~~~~~~~~~~~~~~~~~~~~~~~

As our code uses a special format for the database, the first thing you
need to do is to create a suited structure for the data. You can do this
manually but we give a piece of code to do it automatically which may
prevent bad surprises… It creates validation, training and testing
directories and put a certain number of images per class in it (same
number of image for each class) To do so, you just need to have CG and
PG images in two different directories and choose another directory to
store the formatted database. Then you can just use the *construct\_DB*
method :

.. code:: python

    from CGvsPhoto import construct_DB

    path_CG = '/path/to/CG'
    path_PG = '/path/to/PG'
    path_export = 'path/to/export/database'

    construct_DB(source_real = path_PG, source_CG = path_CG,
                 target_dir = path_export, nb_per_class = 1000,
                 validation_proportion = 0.1, test_proportion = 0.2)

You can choose the total number of images per class and the proportion
of images to put in each directory.

Creating the patches database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our implementation computes local patch classification before
aggregating the results for full-size images. So, to train the
single-image classifier, a patch database must be created. To this end,
use the *Database\_loader* class :

.. code:: python

    from CGvsPhoto import Database_loader

    path_source = 'path/to/source/database'
    path_export = 'path/to/export/patches'
    size_patch = 100

    data = Database_loader(path_source, image_size = size_patch, 
                         only_green=True)

    # export a patch database    
    data.export_database(path_export, 
                         nb_train = 40000, 
                         nb_test = 4000, 
                         nb_validation = 2000)

You can choose the patch size (100x100 pixels in our initial
implementation) and the number of patches to put in each directory (with
50/50 distribution between each class).

Note that supported image extensions are
[“.jpg”,“.gif”,“.png”,“.tga”,“.tif”, “.JPG”, “.jpeg”]

Creating a model
~~~~~~~~~~~~~~~~

Now comes the fun part! In order to create your own model, you just have
to call the *Model* class. For example :

.. code:: python

    from CGvsPhoto import Model

    model = Model(database_path 'Database/My_Patch_Data', image_size = 100,
                  config = 'Config1', filters = [32, 64], 
                  feature_extractor = 'Stats', batch_size = 50)

You can specify the number of output filtered images for each layer with
the parameter ``filters`` and the feature extraction scheme (between
‘Hist’ and ‘Stats’). You also need to give the path to the patch
database.

Warning : The database must contain images with the same image\_size as
specified in parameter image\_size.

Training a classifier
~~~~~~~~~~~~~~~~~~~~~

Now, to train this model, use the *train* function specifying the number
of training/validation/testing batches:

.. code:: python

    model.train(nb_train_batch = 15000,
                nb_test_batch = 80, 
                nb_validation_batch = 40)
                
This will train a model and save the weights and a bunch of summaries in
correspondant directories (you specify the name of the run at the
begining of the procedure). You can also load a pre-trained model and
continue the training (be careful though to load a model which structure
corresponds to the one you are trying to train).

At the end of training, the model’s accuracy is evaluated on the patches
testing set.

Testing
~~~~~~~

Now that you have trained a model, you can load it and test it on
full-size images, using the *test\_total\_images* function :

.. code:: python

    test_data_path = '/Database/My_Data/test/'
    clf.test_total_images(test_data_path = test_data_path,
                          nb_images = 720, decision_rule = 'weighted_vote')

Your test directory must contain two sub-directories : CGG and Real.
Before testing, the console will ask you the name of the weight file to
load. It must be in the default checkpoint directory and you should
inidcate the .ckpt file. You can specify the number of images you want
to process and the aggregation scheme between ‘weighted\_vote’ and
‘majority\_vote’ (even if ‘weighted\_vote’ is in general more
efficient).


Reproducing the results
~~~~~~~

The architecture used in our article [Distinguishing Computer Graphics from Natural Images Using Convolution Neural Networks, WIFS, 2017] can be reproduced by using the default parameters of the Model class. The weights used for this model are available as TensorFlow checkpoints in the folder /weights: use "Stats_15000_run_14800.ckpt" 


Authors
-------

**Nicolas Rahmouni** - `GitHub`_

**Vincent Nozick** - `Website`_

References
-------

Rahmouni, N., Nozick, V., Yamagishi, J., & Echizen, I. (2017, December). Distinguishing Computer Graphics from Natural Images Using Convolution Neural Networks. In IEEE Workshop on Information Forensics and Security, WIFS 2017.

This research was carried out while the authors stayed at the National Institute of Informatics, Japan

.. _GitHub: https://github.com/NicoRahm
.. _state of the art method: http://ieeexplore.ieee.org/abstract/document/6115849/
.. _create\_DB.py: examples/create_DB.py
.. _create\_patches\_splicing.py: examples/create_patches_splicing.py
.. _test\_pipeline.py: examples/test_pipeline.py
.. _test\_splicing.py: examples/test_splicing.py
.. _here: http://www-igm.univ-mlv.fr/~vnozick/publications/Rahmouni_WIFS_2017/Rahmouni_WIFS_2017.pdf
.. _Raise Database: http://mmlab.science.unitn.it/RAISE/
.. _Level-design reference Database: http://level-design.org/referencedb/ 
.. _data directory: https://github.com/NicoRahm/CGvsPhoto/tree/master/data
.. _link: http://www-igm.univ-mlv.fr/~vnozick/publications/Rahmouni_WIFS_2017/GameCG.zip
.. _Website: http://www-igm.univ-mlv.fr/~vnozick/?lang=fr
