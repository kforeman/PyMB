
## Install numpy, scipy and and dependencies on Python side
conda install -c intel -y numpy scipy ipython
conda install -c conda-forge -y scikit-sparse libiconv libxml2 lxml

## Install R and rpy2
conda install -c r -y r-base=3.5.0 
pip install rpy2
