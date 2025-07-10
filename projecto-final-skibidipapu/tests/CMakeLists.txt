add_executable(tensor_test
    test_tensor.cpp
)

target_include_directories(tensor_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_test(NAME TensorTest COMMAND tensor_test) 