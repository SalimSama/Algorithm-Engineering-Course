cmake_minimum_required(VERSION 3.10)
project(quantpng)

# Include subdirectories
add_subdirectory(lodepng)
add_subdirectory(quantize)

# Add the executable
add_executable(quantpng main.cpp)

# Link the executable with the libraries
target_link_libraries(quantpng PUBLIC lodepng quantize)
