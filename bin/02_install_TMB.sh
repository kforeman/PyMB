
#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

## Check R
export PATH="$HOME/miniconda/bin:$PATH"
which -a python
which -a R


# R -e 'install.packages("Matrix",  repos="http://cran.us.r-project.org", dependencies=T)'
# R -e 'install.packages("TMB",  repos="http://cran.us.r-project.org", dependencies=T)'

# cd /opt && \
# git clone https://github.com/kaskr/adcomp.git && cd adcomp && \
# make install-metis-full
