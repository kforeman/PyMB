#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

## Run linreg example
source activate pymb_test
which -a R
python Examples/linreg.py