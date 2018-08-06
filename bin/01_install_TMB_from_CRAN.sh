#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

## Install TMB from CRAN
source activate pymb_test
R -e 'install.packages("TMB",  repos="http://cran.us.r-project.org", dependencies=)'
