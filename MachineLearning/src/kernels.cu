#pragma once

#include <iostream>

#include "kernels.cuh"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
    }

template<int N, int M>
void print_matrix(float A[N][M])
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < M; j++)
    {
      std::cout << A[i][j] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

