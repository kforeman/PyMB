
#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

## Check R
which -a python
echo $PATH
export PATH="$HOME/miniconda/bin:$PATH"
ls ~/

which -a R
R -e 'sessionInfo()'

# R -e 'install.packages("Matrix",  repos="http://cran.us.r-project.org", dependencies=T)'
# R -e 'install.packages("TMB",  repos="http://cran.us.r-project.org", dependencies=T)'

# cd /opt && \
# git clone https://github.com/kaskr/adcomp.git && cd adcomp && \
# make install-metis-full
