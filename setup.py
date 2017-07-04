from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='CGvsPhoto',
      version='0.0.2',
      description='A deep-learning method for distinguishing computer graphics from real photogrphic images',
      long_description=readme(),
      url='https://github.com/NicoRahm/CGvsPhoto',
      author='Nicolas Rahmouni',
      author_email='nicolas.rahmouni@polytechnique.edu',
      packages=find_packages(exclude = ['examples', 'weights']),
      license='MIT',
      include_package_data=True,
      install_requires=['tensorflow>=1.0.1','numpy>=1.6.1',
      					'scikit-learn>=0.18.1', 'Pillow>=3.1.2', 
      					'matplotlib>=1.3.1'],
      zip_safe=False)