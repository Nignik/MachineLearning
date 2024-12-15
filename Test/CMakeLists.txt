file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

add_executable (Test ${SOURCES} ${HEADERS})

set_target_properties(Test PROPERTIES
    LINKER_LANGUAGE CXX
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

find_package(GTest REQUIRED)
find_package(MachineLearning REQUIRED)
target_link_libraries(Test PRIVATE MachineLearning::MachineLearning GTest::gtest GTest::gtest_main)

add_test(NAME MatmulTest COMMAND MatmulTest)

