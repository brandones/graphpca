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
  version = '0.3',
  license = 'Apache License 2.0',
  description = 'Produces a low-dimensional representation of the input graph',
  long_description=(read('README.rst')),
  author = 'Brandon Istenes',
  author_email = 'brandonesbox@gmail.com',
  url = 'https://github.com/brandones/graphpca',
  download_url = 'https://github.com/brandones/graphpca/tarball/0.3',
  keywords = ['graph', 'math', 'pca', 'data', 'reduction', 'dimension', 'compression'],
  classifiers = ['Development Status :: 4 - Beta',
                 'License :: OSI Approved :: Apache Software License']
)
