cmake_minimum_required(VERSION 3.30)

project(Binarize_Algo_Eng VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp")

# Bibliothek erstellen
add_library(image_utils STATIC src/image_utils.cpp)

# Header-Dateien verlinken
target_include_directories(image_utils PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Hauptprogramm erstellen und mit Bibliothek verlinken
add_executable(project src/main.cpp)
target_link_libraries(project PRIVATE image_utils OpenMP::OpenMP_CXX)
