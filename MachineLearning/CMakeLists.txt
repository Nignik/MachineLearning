file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh")

add_library (MachineLearning STATIC ${SOURCES} ${HEADERS})

find_package(CUDAToolkit REQUIRED)
target_include_directories(MachineLearning PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

set_target_properties(MachineLearning PROPERTIES
    LINKER_LANGUAGE CXX
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(MachineLearning PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # During the build
    $<INSTALL_INTERFACE:include>  # After installation
)

# Install the CMake configuration for package finding
install(TARGETS MachineLearning EXPORT MachineLearningConfig 
        ARCHIVE DESTINATION lib           # For static libraries
        LIBRARY DESTINATION lib           # For shared libraries
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

install(EXPORT MachineLearningConfig
    FILE MachineLearningConfig.cmake
    NAMESPACE MachineLearning::
    DESTINATION lib/cmake/MachineLearning
)

install(DIRECTORY include/ DESTINATION include)


