from setuptools import setup, find_packages
import numpy


VERSION = 0.1

setup(
    name='PyMB',
    version=VERSION,
    packages=find_packages(),
    include_dirs=[
        numpy.get_include(),
    ],
    include_package_data=True,
    install_requires=[
        'numpy', 'rpy2',
    ]
)
 
