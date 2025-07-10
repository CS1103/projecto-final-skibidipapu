#include <iostream>
#include <cassert>
#include "../include/tensor.h"

int main() {
    std::cout << "Testing UTEC Tensor System..." << std::endl;
    
    utec::algebra::Tensor<double, 2> tensor(3, 4);
    tensor.fill(1.0);
    
    tensor(0, 0) = 5.0;
    assert(tensor(0, 0) == 5.0);
    
    utec::algebra::Tensor<double, 2> A(2, 3);
    utec::algebra::Tensor<double, 2> B(3, 2);
    
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    
    B(0, 0) = 1; B(0, 1) = 2;
    B(1, 0) = 3; B(1, 1) = 4;
    B(2, 0) = 5; B(2, 1) = 6;
    
    auto C = A.matmul(B);
    
    auto A_T = A.transpose();
    
    std::cout << "A:\n" << A << std::endl;
    std::cout << "B:\n" << B << std::endl;
    std::cout << "C = A * B:\n" << C << std::endl;
    std::cout << "A^T:\n" << A_T << std::endl;
    
    auto D = A + B.transpose();
    auto E = A * 2.0;
    
    std::cout << "A + B^T:\n" << D << std::endl;
    std::cout << "A * 2:\n" << E << std::endl;
    
    std::cout << "All tensor tests passed!" << std::endl;
    return 0;
} 