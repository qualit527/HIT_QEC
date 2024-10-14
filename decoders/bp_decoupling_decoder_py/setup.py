import pybind11
import pybind11.setup_helpers
from setuptools import setup

setup(
    ext_modules=[
        pybind11.setup_helpers.Pybind11Extension(
            name="bpdecoupling",
            sources=["src/main.cpp",
                     "src/bp_decoder.cpp",
                     "src/sparse_matrix.cpp"],
            include_dirs=[pybind11.get_include(), "src/include"],
            language="c++",
            cxx_std="20"
        )
    ]
)
