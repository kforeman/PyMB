#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

ls ~/
export PATH="$HOME/miniconda/bin:$PATH"
which -a python
which -a R

# R -e 'library(TMB); runExample("randomregression");'

