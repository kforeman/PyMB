# PyMB
Python Model Builder - fit statistical models using algorithmic differentiation

## Concept
PyMB uses [algorithmic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation) from 
[CppAD](http://www.coin-or.org/CppAD/) to compute derivatives of objective functions, allowing users to quickly find 
optimal solutions to large, multidimensional statistical models.

## Implementation
#### Current
PyMB is currently just a light wrapper around [Template Model Builder](https://github.com/kaskr/adcomp), allowing users to run 
TMB models directly from Python by transparently passing `numpy` and other numeric data into an embedded `rpy2` instance. 
TMB models are specified using C++ templates, requiring the user to be at least somewhat familiar with C++.

#### Future
We intend to make a fully Pythonic modeling library that closely mirrors [PyMC2](https://github.com/pymc-devs/pymc) in syntax
and ease of model specification, but uses either TMB or [PyCppAD](https://github.com/b45ch1/pycppad) under the hood.

## Example
A demo [iPython notebook](http://ipython.org/notebook.html) can be found at 
[Examples/Linear Regression.ipynb](Examples/Linear Regression.ipynb) or an online version can be viewed [here]
(http://nbviewer.ipython.org/github/kforeman/PyMB/blob/master/Examples/Linear%20Regression.ipynb).

## Installation
#### Dependencies
* [R](http://www.r-project.org/)
* [TMB](https://github.com/kaskr/adcomp)
* [rpy2](http://rpy.sourceforge.net/)
* [numpy](http://www.numpy.org/)

#### Importing
`setup.py` has not yet been added. For now simply `git clone git@github.com:kforeman/PyMB.git` into your working directory, 
startup Python or an iPython notebook, and `import PyMB`.
