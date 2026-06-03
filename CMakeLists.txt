cmake_minimum_required(VERSION 3.16)
project(VectorForge)

# Use C++17
set(CMAKE_CXX_STANDARD 17)

# Create library
add_library(vectorforge
    src/distance.cpp
)

# Include headers
target_include_directories(vectorforge PUBLIC include)

# AVX2 + FMA flags (IMPORTANT)
if (MSVC)
    target_compile_options(vectorforge PUBLIC /arch:AVX2)
else()
    target_compile_options(vectorforge PUBLIC -mavx2 -mfma)
endif()

# GOOGLE TEST 
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

# Create test executable
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic vectorforge gtest_main)
add_test(NAME basic_test COMMAND test_basic)

# Create distance test executable
add_executable(test_distance tests/test_distance.cpp)
target_link_libraries(test_distance vectorforge gtest_main)
add_test(NAME distance_test COMMAND test_distance)