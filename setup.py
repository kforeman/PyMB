from setuptools import setup
import numpy

setup(
    name='PyMB',
    version='0.1',
    packages=['', 'PyMB.magic'],
    url='https://github.com/kforeman/PyMB',
    license='GNU General Public License v2.0',
    author='Kyle Foreman',
    author_email='kforeman <at> post <dot> harvard <dot> edu',
    description='Python Model Builder',
    include_dirs=[
        'numpy', 'rpy2', 'scipy',
    ]
)
