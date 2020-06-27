#!/bin/bash

cd "$(dirname "$0")"
./env/bin/black graphpca/*.py test/*.py
