# Computer Graphics vs Real Photographic Images : A Deep-learning approach

**CGvsPhoto** implements a method for computer graphics detection using Convolutional Neural Networks with TensorFlow back-end.

![splicing](https://user-images.githubusercontent.com/17125992/26874001-e4716e50-4bb6-11e7-929d-a7f2e7192a9f.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes. 

### Prerequisites
* Python 3.6+
* Numpy
* Scikit-learn 
* TensorFlow 1.0.1+ (https://github.com/tensorflow/tensorflow)
* PIL
* Matplotlib


### Installing

Simply clone the repository into your favorite directory.

```
git clone https://github.com/NicoRahm/CGvsPhoto
```
Then, when you are on the directory, you can install the package using : 
```
pip install .
```

To run the first test, there is one more thing to set up : 
* Create a file named config.ini in your execution directory to store the different configurations of your environment. The format is the following :
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
You can create it manually or using the function construct

### Some simple examples

To get started, you can run simple scripts from the examples directory. Do not forget to set up the config.ini file correctly as described above.

* create_DB.py will create a formated database for future tests.
* test_pipeline.py trains a neural network to classify image patches and then evaluate it.
* ...

## How to use

This section explains basic uses of this code. 

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

