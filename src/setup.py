from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("cmodule/biased_error_equation.pyx"),
    include_dirs=[np.get_include()],
)
