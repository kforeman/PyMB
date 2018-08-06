#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

## Install PyMB
cd ../PyMB
python setup.py install
