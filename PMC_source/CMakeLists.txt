cmake_minimum_required(VERSION 3.9)
project(projection_memory_clustering)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -O3 -std=c++14 -pthread")
set(CMAKE_BUILD_TYPE Release)

set(SOURCE_FILES main.cpp Cluster.cpp ContainerClusters.cpp subspaceClustering.h)

set(OUTPUT_FILE "PMC")

add_executable(${OUTPUT_FILE} ${SOURCE_FILES})
target_link_libraries(${OUTPUT_FILE} pthread)

