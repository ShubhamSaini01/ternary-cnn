cmake_minimum_required(VERSION 3.14)
project(ternary_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AVX2 + FMA
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -O3 -march=native")

add_executable(ternary_inference
    src/main.cpp
)

target_include_directories(ternary_inference PRIVATE src)
