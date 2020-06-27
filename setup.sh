#!/usr/bin/env bash

python3 -m venv env
./env/bin/pip install --upgrade pip
./env/bin/pip install -r requirements.txt
ln -s ../../hooks/pre-commit .git/hooks/