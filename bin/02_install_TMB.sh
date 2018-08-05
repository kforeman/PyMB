
R -e 'install.packages("Matrix",  repos="http://cran.us.r-project.org", dependencies=T)'
cd /opt && \
git clone https://github.com/kaskr/adcomp.git && cd adcomp && \
make install-metis-full
