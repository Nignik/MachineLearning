cmake_minimum_required (VERSION 3.17)

project(MachineLearning LANGUAGES CXX CUDA)
#enable_language(CUDA)

#set(GTest ${CMAKE_SOURCE_DIR}/external/googletest)

#add_subdirectory("external")
add_subdirectory("MachineLearning")
add_subdirectory("Test")
