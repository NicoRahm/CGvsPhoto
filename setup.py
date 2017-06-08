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
      zip_safe=False)