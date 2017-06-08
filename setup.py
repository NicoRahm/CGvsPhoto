from setuptools import setup
from setuptools import find_packages

setup(name='CGvsPhoto',
      version='0.1',
      description='A deep-learning method for distinguishing computer graphics from real photogrphic images',
      url='https://github.com/NicoRahm/CGvsPhoto',
      author='Nicolas Rahmouni',
      author_email='nicolas.rahmouni@polytechnique.edu',
      packages=find_packages(exclude = ['examples']),
      license='MIT',
      install_requires=['tensorflow>=1.0.1','numpy>=1.6.1',
      					'scikit-learn>=0.18.1', 'Pillow>=3.1.2', 
      					'matplotlib>=1.3.1'],
      zip_safe=False)