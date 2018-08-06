#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

which -a python
R -e 'library(TMB); runExample("randomregression");