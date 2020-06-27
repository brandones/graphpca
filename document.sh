#!/bin/bash

cd docs_src
make html
cd ..
cp -r docs_src/build/html docs