#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

## Build TMB from source
source activate pymb_test
git clone https://github.com/kaskr/adcomp.git && \
    cd adcomp && \
    make install-metis-full && cd ..