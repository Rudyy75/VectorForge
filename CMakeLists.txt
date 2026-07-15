cmake_minimum_required(VERSION 3.16)
project(VectorForge)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VECTORFORGE_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

if(VECTORFORGE_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_compile_definitions(VECTORFORGE_HAS_CUDA)
endif()

# Create library
add_library(vectorforge
    src/distance.cpp
    src/flat_index.cpp
    src/thread_pool.cpp
    src/kmeans.cpp
    src/ivf_index.cpp
    src/compute_backend.cpp
)
set_target_properties(vectorforge PROPERTIES OUTPUT_NAME "vectorforge_core")

if(VECTORFORGE_ENABLE_CUDA)
    # The CUDA backend library
    add_library(vectorforge_cuda STATIC
        src/cuda/distance_kernels.cu
        src/cuda/knn_kernels.cu
    )
    if(MSVC)
        target_compile_options(vectorforge_cuda PRIVATE -Xcompiler=/Zc:preprocessor)
    endif()
    target_include_directories(vectorforge_cuda PUBLIC include)
    
    # Use cuBLAS for matrix operations
    target_link_libraries(vectorforge_cuda PRIVATE cublas)
    
    # Link the CUDA backend into the main library
    target_link_libraries(vectorforge PUBLIC vectorforge_cuda)
endif()

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
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)

set(PYBIND11_FINDPYTHON ON)
FetchContent_MakeAvailable(googletest pybind11)

enable_testing()

# Create test executable
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic vectorforge gtest_main)
add_test(NAME basic_test COMMAND test_basic)

# Create distance test executable
add_executable(test_distance tests/test_distance.cpp)
target_link_libraries(test_distance vectorforge gtest_main)
add_test(NAME distance_test COMMAND test_distance)

# Create flat index test executable
add_executable(test_flat_index tests/test_flat_index.cpp)
target_link_libraries(test_flat_index vectorforge gtest_main)
add_test(NAME flat_index_test COMMAND test_flat_index)

# Create thread pool test executable
add_executable(test_thread_pool tests/test_thread_pool.cpp)
target_link_libraries(test_thread_pool vectorforge gtest_main)
add_test(NAME thread_pool_test COMMAND test_thread_pool)

# Create kmeans test executable
add_executable(test_kmeans tests/test_kmeans.cpp)
target_link_libraries(test_kmeans vectorforge gtest_main)
add_test(NAME kmeans_test COMMAND test_kmeans)

# Create ivf index test executable
add_executable(test_ivf_index tests/test_ivf_index.cpp)
target_link_libraries(test_ivf_index vectorforge gtest_main)
add_test(NAME ivf_index_test COMMAND test_ivf_index)

# Create persistence test executable
add_executable(test_persistence tests/test_persistence.cpp)
target_link_libraries(test_persistence vectorforge gtest_main)
add_test(NAME persistence_test COMMAND test_persistence)

# Python Bindings
pybind11_add_module(vectorforge_py python/bindings.cpp)
target_link_libraries(vectorforge_py PRIVATE vectorforge)
set_target_properties(vectorforge_py PROPERTIES OUTPUT_NAME "vectorforge")

if(MINGW)
    target_link_options(vectorforge_py PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive)
endif()

# Fetch JSON and Crow
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

FetchContent_Declare(asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG        asio-1-28-0
)
FetchContent_MakeAvailable(asio)
set(ASIO_INCLUDE_DIR ${asio_SOURCE_DIR}/asio/include CACHE PATH "" FORCE)

set(CROW_FETCH_ASIO ON CACHE BOOL "" FORCE)
set(CROW_FEATURES "asio" CACHE STRING "" FORCE)
FetchContent_Declare(crow
    GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
    GIT_TAG        v1.2.0
)

FetchContent_MakeAvailable(json crow)

# Create Server Executable
add_executable(vectorforge_server api/server.cpp)
target_link_libraries(vectorforge_server PRIVATE vectorforge nlohmann_json::nlohmann_json Crow::Crow)
if(MINGW OR WIN32)
    target_link_libraries(vectorforge_server PRIVATE ws2_32 mswsock)
    if(MINGW)
        target_link_options(vectorforge_server PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive)
    endif()
endif()