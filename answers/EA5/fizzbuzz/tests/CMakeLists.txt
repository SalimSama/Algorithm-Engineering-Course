# Include FetchContent module
include(FetchContent)

# Declare the FetchContent for Catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1 # latest stable release
)

# Make Catch2 available
FetchContent_MakeAvailable(Catch2)

# Create the test executable
add_executable(test_fizzbuzz test_fizzbuzz.cpp)

# Link the fizzbuzz library and Catch2 to the test executable
target_link_libraries(test_fizzbuzz PRIVATE fizzbuzz Catch2::Catch2WithMain)

# Include CTest and Catch2 modules
include(CTest)
include(Catch)

# Discover and register tests with CTest
catch_discover_tests(test_fizzbuzz)
