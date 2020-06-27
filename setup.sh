#!/usr/bin/env bash

python3 -m venv env
./env/bin/pip install --upgrade pip
./env/bin/pip install -r requirements.txt

rm -rf .git/hooks/pre-commit
ln -s ../../hooks/pre-commit .git/hooks/

rm -rf docs/output/*
for f in $(ls -d output/*); do ln -s ../../$f docs/output/; done
