cmake_minimum_required (VERSION 3.15)

project(TestBenchmark CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(benchmark REQUIRED)

add_executable(searches searches.cpp)
target_link_libraries(searches PRIVATE benchmark::benchmark)

