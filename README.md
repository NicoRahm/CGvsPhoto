# Computer Graphics vs Real Photographic Images : A Deep-learning approach

**CGvsPhoto** implements a method for computer graphics detection using Convolutional Neural Networks with TensorFlow back-end.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

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

To run this code from a distant directory, you have to set up two things : 
* Add the path to the CGvsPhoto directory.
```python
>>> import sys
>>> sys.path.append('../path/to/CGvsPhoto')
```
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


### Some simple examples

To get started, you can run simple scripts from the examples directory. Do not forget to set up the config.ini file correctly as described above.

* test_pipeline.py trains a neural network to classify image patches.
* 

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

## Acknowledgments

