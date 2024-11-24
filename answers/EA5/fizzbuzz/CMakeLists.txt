cmake_minimum_required(VERSION 3.10)
project(FizzBuzzLib LANGUAGES CXX)

# Enable testing
enable_testing()

# Add the library directory
add_subdirectory(src)

# Add the tests directory
add_subdirectory(tests)
