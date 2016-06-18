#!/usr/bin/env bash

virtualenv env
./env/bin/pip install --upgrade pip
./env/bin/pip install -r requirements.txt
