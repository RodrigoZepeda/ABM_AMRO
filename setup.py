import numpy
from distutils.core import setup, Extension

# OSX build with CC=/usr/local/opt/llvm/bin/clang python setup.py build_ext -i
# after installing brew install llvm libomp
cpp_args = ['-std=c++14', '-O3', '-Wall']
ext_modules = [
    Extension(
        'amro',
        ['abm_wards.cpp', 'amro.cpp'],
        include_dirs=["armadillo_12_6_1",
                      'pybind11/include',
                      numpy.get_include(),
                      "carma_0_6_7"],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='amro',
    version='0.0.1',
    author='Rodrigo Zepeda-Tello',
    license='MIT',
    author_email='rodrigo.zepeda@columbia.edu',
    description='Agent Based Model for Antimicrobial Resistance',
    ext_modules=ext_modules,
    setup_requires=['numpy','pybind11'],
    install_requires=['pybind11', 'numpy'],  # external packages as dependencies
)
