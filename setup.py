import numpy
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import sys

__version__ = "0.0.2"

# OSX build with CC=/usr/local/opt/llvm/bin/clang python setup.py build_ext -i
# after installing brew install llvm libomp
cpp_args = ['-fopenmp']
if sys.platform == "darwin":
    cpp_args.insert(0,"-Xpreprocessor") 

ext_modules = [
    Pybind11Extension(
        'amro',
        ['src/amro/abm_wards.cpp', 'src/amro/amro.cpp'],
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs=["armadillo_12_6_1",
                      'pybind11/include',
                      numpy.get_include(),
                      "carma_0_6_7"],
        language='c++',
        extra_compile_args=cpp_args,
        extra_link_args=["-lgomp"],
    ),
]

setup(
    name='amro',
    version=__version__,
    author='Rodrigo Zepeda-Tello',
    license='MIT',
    author_email='rodrigo.zepeda@columbia.edu',
    description='Agent Based Model for Antimicrobial Resistance',
    ext_modules=ext_modules,
    zip_safe=False, #Historical flag https://setuptools.pypa.io/en/latest/deprecated/zip_safe.html
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
)
