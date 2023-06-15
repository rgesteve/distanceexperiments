cmake_minimum_required (VERSION 3.15)

project(TestBenchmark CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(benchmark REQUIRED)

add_executable(searches searches.cpp)
target_link_libraries(searches PRIVATE benchmark::benchmark)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sapphirerapids -O2 -mprefer-vector-width=512")
endif()

#add_executable(dot_autovec testFP16_autovec.cpp)
#add_executable(dot_bf16 testBF16.cpp)
add_executable(dot_fp16 tfp16.cpp)
add_executable(dotprod driver.cpp testFP16.cpp testBF16.cpp)
