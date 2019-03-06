from setuptools import setup, find_packages
import numpy
VERSION = 0.1

interactive_requirements = [
    'IPython',
    'jupyter',
]


setup(
    name='PyMB',
    version=VERSION,
    packages=find_packages(),
    url='https://github.com/kforeman/PyMB',
    license='GNU General Public License v2.0',
    author='Kyle Foreman',
    author_email='kforeman <at> post <dot> harvard <dot> edu',
    description='Python Model Builder',
    include_dirs=[
        numpy.get_include(),
    ],
    install_requires=[
        'numpy', 'rpy2<=2.9.5', 'scipy', 'scikit-sparse',
    ],
    extras_require={
        'interactive': interactive_requirements,
    },
    classifiers=(
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: R",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v2.0",
        "Natural Language :: English",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering",
    ),
)
