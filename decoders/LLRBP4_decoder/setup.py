# setup.py
from setuptools import setup, Extension
from setuptools import find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

# 获取 Eigen 的包含路径
# 假设 Eigen 安装在系统路径中，如果在自定义路径，请修改下面的路径
eigen_include = 'eigen'  # 修改为您的 Eigen 头文件路径

ext_modules = [
    Pybind11Extension(
        "LLRBP4_decoder",
        ["src/LLRBp4Decoder.cpp", "src/decoder_bindings.cpp"],
        include_dirs=[
            pybind11.get_include(),
            eigen_include,  # 添加 Eigen 头文件路径
            "include"       # 添加您的头文件路径
        ],
        language='c++',
        extra_compile_args=["/Od", "/Zi"],
    ),
]

setup(
    name="LLRBP4_decoder",
    version="0.1.0",
    author='Jiahan Chen',
    author_email='jiahanchen527@gmail.com',
    description='Log-likelihood quaternary domain belief propagation decoder (and its variations) for quantum error correction codes',
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=["pybind11", "numpy"],
    packages=find_packages(),
)
