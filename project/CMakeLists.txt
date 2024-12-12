cmake_minimum_required(VERSION 3.30)

# Projektname und Version
project(project VERSION 1.0)

# Setze den C++-Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Füge die Quelldateien hinzu
add_executable(project main.cpp)

# Inkludiere STB-Header-Dateien
target_include_directories(project PRIVATE ${CMAKE_SOURCE_DIR})

# Erstelle ein Custom Command für das Projekt





