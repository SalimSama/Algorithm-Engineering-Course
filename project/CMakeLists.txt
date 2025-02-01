cmake_minimum_required(VERSION 3.29)

project(Binarize_Algo_Eng VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -ffast-math -funroll-loops")
add_subdirectory(external/spdlog)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Hauptprogramm erstellen und mit Bibliothek verlinken
add_executable(project
        src/main.cpp
        src/binarization/thresholding.cpp
        src/binarization/adaptive_thresholding.cpp
        src/binarization/integral_binarization.cpp
        src/filters/adaptive_median_filter.cpp
        src/utils/image_io.cpp
        src/utils/stb_image_implementation.cpp
)

target_link_libraries(project OpenMP::OpenMP_CXX spdlog)
