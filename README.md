# Computer Graphics vs Real Photographic Images : A Deep-learning approach

**CGvsPhoto** implements a method for computer graphics detection using Convolutional Neural Networks with TensorFlow back-end. The package contains methods for extracting patches from computer graphics and real images, training a CNN with a custom statistical layer, testing this model, compare with a [state of the art method](http://ieeexplore.ieee.org/abstract/document/6115849/), visualizing probability maps, etc.

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
    training/
        CGG/
        Real/
    validation/ 
        CGG/
        Real/
    testing/
        CGG/
        Real/
```
You can create it manually or using the function construct_DB. 

### Some simple examples

To get started, you can run simple scripts from the examples directory. Do not forget to set up the config.ini file correctly as described above.

* [create_DB.py](examples/create_DB.py) will create a formated database for future tests.
* [test_pipeline.py](examples/test_pipeline.py) trains a neural network to classify image patches and then evaluate it.
* ...

## How to use

This section explains basic uses of this code. 

### Creating a model

To create your own model, you just have to call the Model class. For example : 
```python
from CGvsPhoto import Model

model = Model(database_path 'Database/My_Data', image_size = 100,
              config = 'Config1', filters = [32, 64], 
              feature_extractor = 'Stats', batch_size = 50)
```



### Training a classifier


```
Give an example
```

### Testing


```
Give an example
```

## Authors

**Nicolas Rahmouni**  - [NicoRahm](https://github.com/NicoRahm)

**Vincent Nozick**

