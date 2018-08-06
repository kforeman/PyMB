FROM continuumio/miniconda3

## Get all the Linux dependencies
RUN apt-get update --fix-missing && \
    apt-get install --yes \
        wget \
        curl \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        git \
        gcc \
        g++ \
        libreadline-dev \
        libeigen3-dev \
        tar \
        build-essential && \
    apt-get purge && \
    apt-get clean

## Install Python dependencies
## NOTE: DO NOT install rpy2 from conda-forge, got issues
RUN conda install -c conda-forge -y numpy scipy && \
    scikit-sparse libiconv libxml2 lxml

## Install R and rpy2
RUN conda install -c r -y \
    r-base=3.5.0 && \
    pip install rpy2


## Build TMB and it's dependencies
RUN R -e 'install.packages(c("Matrix", "numDeriv", "RcppEigen"),  repos="http://cran.us.r-project.org", dependencies=T)' && \
    cd /opt && \
    git clone https://github.com/kaskr/adcomp.git && cd adcomp && \
    make install-metis-full

## Install PyMB
RUN cd /opt && \
    git clone https://github.com/kforeman/PyMB.git && \
    cd PyMB && python setup.py install