# setup.py
#

import os

from setuptools import setup

def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

setup(
  name = 'graphpca',
  packages = ['graphpca'],
  version = '1.0.0',
  license = 'Apache License 2.0',
  description = 'Produces a low-dimensional representation of the input graph',
  long_description=(read('README.rst')),
  long_description_content_type='text/x-rst',
  author = 'Brandon Istenes',
  author_email = 'brandonesbox@gmail.com',
  install_requires=[
    'networkx',
    'numpy',
    'scipy'
  ],
  url = 'https://github.com/brandones/graphpca',
  download_url = 'https://github.com/brandones/graphpca/tarball/0.5',
  keywords = ['graph', 'draw', 'pca', 'data', 'reduction', 'dimension', 'compression'],
  classifiers = ['Development Status :: 4 - Beta',
                 'License :: OSI Approved :: Apache Software License']
)
