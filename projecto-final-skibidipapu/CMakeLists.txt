cmake_minimum_required(VERSION 3.18)
project(UTEC_NeuralNetwork CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    src/main.cpp
)

set(HEADERS
    include/tensor.h
    include/nn_interfaces.h
    include/nn_activation.h
    include/nn_dense.h
    include/nn_loss.h
    include/nn_optimizer.h
    include/neural_network.h
)

add_executable(neural_net_demo ${SOURCES} ${HEADERS})

target_include_directories(neural_net_demo PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()
add_subdirectory(tests)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neural_net_demo PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(neural_net_demo PRIVATE -O3)
endif()

if(MINGW)
    target_compile_options(neural_net_demo PRIVATE -fopenmp)
    target_link_libraries(neural_net_demo PRIVATE -fopenmp)
endif()

message(STATUS "UTEC Neural Network Project")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}") 