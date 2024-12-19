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

template<int N, int M>
__global__ void forwardKernel(float* X, float* W, float* B, float* Y) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < M) {
		Y[col] = B[col];
		for (int i = 0; i < N; i++) {
			Y[col] += X[i] * W[i * M + col];
		}
	}
}

template<int N, int M>
void forward(float* X, float* W, float* B, float* Y) {
	constexpr int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float* d_X, * d_W, * d_B, *d_Y;
	CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_W, M * N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_B, M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_Y, M * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_W, W, M * N * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, B, M * sizeof(float), cudaMemcpyHostToDevice));

	forwardKernel<N, M><<<numBlocks, threadsPerBlock>>> (d_X, d_W, d_B, d_Y);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(Y, d_Y, M * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_X);
	cudaFree(d_W);
	cudaFree(d_B);
	cudaFree(d_Y);
}

template<int N, int M>
__global__ void reluKernel(float* X, float* Y) {
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < N && col < M) {
		Y[M * row + col] = X[M * row + col] >= 0 ? X[M * row + col] : 0;
	}
}

template<int N, int M>
void relu(float* X, float* Y) {
	constexpr int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float* d_X, * d_W, * d_B, *d_Y;
	CUDA_CHECK(cudaMalloc(&d_X, N * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_Y, N * M * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_X, X, N * M * sizeof(float), cudaMemcpyHostToDevice));

	reluKernel<N, M><<<numBlocks, threadsPerBlock>>> (d_X, d_Y);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(Y, d_Y, N * M * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_X);
	cudaFree(d_Y);
}