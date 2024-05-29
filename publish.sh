#!/bin/bash

# fail on non-zero exit code
set -e

twine check dist/*

twine upload --skip-existing dist/*