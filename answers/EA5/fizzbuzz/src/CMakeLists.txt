# Create a static library from fizzbuzz.cpp
add_library(fizzbuzz STATIC fizzbuzz.cpp)

# Specify the include directories for the library
target_include_directories(fizzbuzz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
