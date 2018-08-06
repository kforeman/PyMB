#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

## Install Miniconda3
# sudo apt-get install -y \
#     wget \
#     curl \
#     bzip2 \
#     ca-certificates \
#     libglib2.0-0 \
#     libxext6 \
#     libsm6 \
#     libxrender1 \
#     git \
#     gcc \
#     g++ \
#     libreadline-dev \
#     tar \
#     build-essential
# wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
# bash miniconda.sh -b -p $HOME/miniconda3
# export PATH="$HOME/miniconda3/bin:$PATH"

## Install numpy, scipy and and dependencies on Python side
# conda install -c conda-forge  -y numpy scipy libiconv libxml2 lxml scikit-sparse

## Install rpy2
# pip install rpy2

## Check R and Python versions
which -a python
which -a R