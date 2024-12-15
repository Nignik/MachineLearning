#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
    }

template<int N, int M, int P>
__global__ void matmulKernel(float* A, float* B, float* C)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < P)
	{
		float dot_prod = 0.0f;
		for (int i = 0; i < M; i++)
		{
			dot_prod += A[row * M + i] * B[i * P + col];
		}
		C[row * P + col] = dot_prod;
	}
}


template<int N, int M, int P>
void matmul(float* A, float* B, float* C)
{
	constexpr int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((P + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float* d_A, * d_B, * d_C;
	CUDA_CHECK(cudaMalloc(&d_A, N * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_B, M * P * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_C, N * P * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, B, M * P * sizeof(float), cudaMemcpyHostToDevice));

	matmulKernel<N, M, P><<<numBlocks, threadsPerBlock>>> (d_A, d_B, d_C);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(C, d_C, N * P * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
