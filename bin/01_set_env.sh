
## Install numpy, scipy and and dependencies on Python side
conda install -y numpy scipy libiconv libxml2 lxml
conda install -c conda-forge -y scikit-sparse

## Install R and rpy2
conda install -c r -y r-base=3.5.0
pip install rpy2
