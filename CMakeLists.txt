cmake_minimum_required (VERSION 3.15)

project(TestBenchmark CXX)
set(CMAKE_CXX_STANDARD 20)

# find_package(SQLite3 REQUIRED)

add_executable(searches searches.cpp)
#target_link_libraries(testConan SQLite::SQLite3)
