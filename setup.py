# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) 2021, Radmann/Jekosch
#------------------------------------------------------------------------------
from setuptools import setup
from os.path import join, abspath, dirname
import os


bf_version = "22.01"
bf_author = "Radmann/Jekosch"

# Get the long description from the relevant file
here = abspath(dirname(__file__))
with open(join(here, 'README.rst')) as f:
    long_description = f.read()


install_requires = list([
	'numpy<1.21',
	'numba',
	'traitlets',	
        'setuptools',
	])

setup_requires = list([
	'numpy<1.21',
	'numba',
	'traitlets',	
        'setuptools',
	])

setup(name="placeholder", 
      version=bf_version, 
      description="Library for calculating Silencers",
      long_description=long_description,
      license="BSD",
      author=bf_author,
      author_email="vincent.radmann@tu-berlin.de",
      classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Physics',
      'License :: OSI Approved :: BSD License',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      ],
      keywords='acoustic silencers plates ',
      packages = ['placeholder'],

      install_requires = install_requires,

      setup_requires = setup_requires,
      
      include_package_data = True,
      #to solve numba compiler 
      zip_safe=False
)

