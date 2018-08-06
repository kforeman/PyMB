
## Script: run linear regression, using a subprocess to build the model
import numpy as np
import os
import argparse


import PyMB
import rpy2.robjects as ro
import rpy2.rinterface as rin


if __name__ == '__main__':

    # Build the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_eigen_inc", help="Add Eigen library to build", required=True)
    parser.add_argument(
        "--use_native_compiler", help="Use R subprocess to compile?", required=True)
    args = parser.parse_args()
    eig_use = args.use_eigen_inc
    use_nat_comp = args.use_native_compiler

    # create an empty model
    m = PyMB.model(name='model1')

    # code for a simple linear regression
    model1 = '''
    #include <TMB.hpp>
    template<class Type>
    Type objective_function<Type>::operator() (){
    // DATA
      DATA_VECTOR(Y);
      DATA_VECTOR(x);

    // PARAMETERS
      PARAMETER(alpha);
      PARAMETER(Beta);
      PARAMETER(logSigma);

    // MODEL
      vector<Type> Y_hat = alpha + Beta*x;
      REPORT(Y_hat);
      Type nll = -sum(dnorm(Y, Y_hat, exp(logSigma), true));
      return nll;
    }
    '''

    # Get the necessary paths to compile TMB code
    tmbinclude = ro.r('paste0(find.package("TMB"), "/include")')[0]
    if use_nat_comp:
      eigeninclude = ro.r('paste0(find.package("RcppEigen"), "/include")')[0]
      tmbinclude = tmbinclude + ' -I' + eigeninclude
    rinclude = rin.R_HOME + "/include"
    rlib = rin.R_HOME + "/lib"

    # rpy2.rinterface.R_HOME

    # compile the model
    m.compile(codestr=model1,
              cc='g++',
              R=rinclude,
              TMB=tmbinclude,
              LR=rlib,
              verbose=True,
              load=True,
              use_native_compiler=use_nat_comp)

    # simulate data
    N = 100
    m.data['x'] = np.arange(N)
    m.data['Y'] = m.data['x'] + 0.5 + np.random.rand(N)

    # set initial parameter values
    m.init['alpha'] = 0.
    m.init['Beta'] = 0.
    m.init['logSigma'] = 0.

    # set random parameters
    m.random = ['alpha', 'Beta']

    # fit the model
    m.optimize(draws=500)
    print(m.report('Y_hat'))
    m.print_parameters()
