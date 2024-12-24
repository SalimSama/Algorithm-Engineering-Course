cmake_minimum_required(VERSION 3.30)

project(Binarize_Algo_Eng VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Quelldateien
add_executable(project main.cpp image_utils.cpp)

# Include-Directory
target_include_directories(project PRIVATE ${CMAKE_SOURCE_DIR})
