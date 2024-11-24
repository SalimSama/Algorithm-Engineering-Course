cmake_minimum_required(VERSION 3.10)

# Create a library for lodepng
add_library(lodepng STATIC lodepng.cpp)

# Include directory for lodepng headers
target_include_directories(lodepng PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

