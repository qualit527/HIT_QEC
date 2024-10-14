# CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(LLRBP4_decoder LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 pybind11
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# 添加 include 目录
include_directories(include)

# 创建 pybind11 模块
pybind11_add_module(decoder src/decoder_bindings.cpp src/LLRBp4Decoder.cpp)

# 链接 Eigen 库
target_link_libraries(decoder PRIVATE Eigen3::Eigen)
