#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

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

/*
* Takes in two input matrices and one output matrix.
* A - NxM input matrix
* B - MxP input matrix
* C - NxP output matrix
*/
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

/*
* Takes in three input matrices and one output matrix
* X - input Nx1 matrix
* W - weights NxM matrix
* B - bias Mx1 matrix
* Y - output Mx1 matrix
*/
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

/*
* Takes in two matrices
* X - NxM input matrix
* Y - NxM output matrix
*/
template<int N, int M>
void relu(float* X, float* Y) {
	constexpr int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float *d_X, *d_Y;
	CUDA_CHECK(cudaMalloc(&d_X, N * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_Y, N * M * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_X, X, N * M * sizeof(float), cudaMemcpyHostToDevice));

	reluKernel<N, M><<<numBlocks, threadsPerBlock>>> (d_X, d_Y);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(Y, d_Y, N * M * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_X);
	cudaFree(d_Y);
}


template<int N, int M>
__global__ void softmaxKernel(float* X, float* Y)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < N && col < M)
	{
		float mx = -FLT_MAX;
		for (int i = 0; i < M; i++)
		{
			mx = max(mx, X[row * M + i]);
		}

		float sum = 0.f;
		for (int i = 0; i < M; i++)
		{
			sum += expf(X[row * M + i] - mx);
		}

		Y[row * M + col] = expf(X[row * M + col] - mx) / sum;
	}
}

/*
* Takes in two matrices
* X - NxM input matrix
* Y - NxM output matrix
*/
template<int N, int M>
void softmax(float* X, float* Y)
{
	constexpr int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float *d_X, *d_Y;
	CUDA_CHECK(cudaMalloc(&d_X, N * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_Y, N * M * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_X, X, N * M * sizeof(float), cudaMemcpyHostToDevice));

	softmaxKernel<N, M> <<<numBlocks, threadsPerBlock >>> (d_X, d_Y);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(Y, d_Y, N * M * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_X);
	cudaFree(d_Y);
}

template<int N, int M>
__global__ void crossEntropyKernel(float* P, float* Q, float* H)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < N)
	{
		float sum = 0.f;
		for (int i = 0; i < M; i++)
		{
			sum -= Q[row * M + i] * logf(fmaxf(1e-6f, P[row * M + i]));
		}
		H[row] = sum;
	}
}

/*
* Takes in tree matrices
* P - NxM probabilities input matrix
* Q - NxM expected probabilities input matrix
* H - N dimensional output vector
*/
template<int N, int M>
void crossEntropy(float* P, float* Q, float* H)
{
	constexpr int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float* d_P, * d_Q, *d_H;
	CUDA_CHECK(cudaMalloc(&d_P, N * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_Q, N * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_H, N * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_P, P, N * M * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_Q, Q, N * M * sizeof(float), cudaMemcpyHostToDevice));

	crossEntropyKernel<N, M> << <numBlocks, threadsPerBlock >> > (d_P, d_Q, d_H);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(H, d_H, N * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_P);
	cudaFree(d_Q);
	cudaFree(d_H);
}


template<int N, int M>
__global__ void initRandomKernel(float* Y)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < N && col < M)
	{
		curandState state;
		curand_init(42, row * M + col, 0, &state);
		Y[row * M + col] = curand_normal(&state) * sqrtf(2.f / N);
	}
}

/*
* Takes in one matrix
* Y - NxM output matrix
*/
template<int N, int M>
void initRandom(float* Y)
{
	constexpr int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float *d_Y;
	CUDA_CHECK(cudaMalloc(&d_Y, N * M * sizeof(float)));

	initRandomKernel<N, M> << <numBlocks, threadsPerBlock >> > (d_Y);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(Y, d_Y, N * M * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_Y);
}


template<int N, int M>
__global__ void crossEntropyBackwardsKernel(float* prediction, float* real, float* output)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (col < M && row < N) {
		output[row * M + col] = prediction[row * M + col] - real[row * M + col];
	}
}

/*
* Takes in two matrices
* prediction - Matrix of size N representing softmax results of the layer
* real - Matrix of size N representing probabilities of the layer
* output - The result
*/
template<int N, int M>
void crossEntropyBackwards(float* prediction, float* real, float* output)
{
	constexpr int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float *d_prediction, *d_real, *d_output;
	CUDA_CHECK(cudaMalloc(&d_prediction, N * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_real, N * M * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_output, N * M * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_prediction, prediction, N * M * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_real, real, N * M * sizeof(float), cudaMemcpyHostToDevice));

	crossEntropyBackwardsKernel<N, M> << <numBlocks, threadsPerBlock >> > (d_prediction, d_real, d_output);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(output, d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_prediction);
	cudaFree(d_real);
	cudaFree(d_output);
}

template<int Batch, int InFeatures, int OutFeatures>
__global__ void backwardKernel(float* weights, float* gradients, float* outputs)
{
	int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int sample_idx = blockIdx.y * blockDim.y + threadIdx.y;

	if (in_idx < InFeatures && sample_idx < Batch)
	{
		float dot = 0.f;
		for (int out_idx = 0; out_idx < OutFeatures; ++out_idx)
		{
			dot += weights[in_idx * OutFeatures + out_idx] * gradients[sample_idx * OutFeatures + out_idx];
		}
		outputs[sample_idx * InFeatures + in_idx] = dot;
	}
}

template<int Batch, int InFeatures, int OutFeatures>
void backward(float* weights, float* loss_gradient, float* output)
{
	constexpr int BLOCK_SIZE = 16;
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((OutFeatures + BLOCK_SIZE - 1) / BLOCK_SIZE, (Batch + BLOCK_SIZE - 1) / BLOCK_SIZE);

	float* d_weights, * d_loss_gradient, * d_output;
	CUDA_CHECK(cudaMalloc(&d_weights, InFeatures * OutFeatures * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_loss_gradient, Batch * OutFeatures * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_output, Batch * InFeatures * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_weights, weights, InFeatures * OutFeatures * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_loss_gradient, loss_gradient, Batch * OutFeatures * sizeof(float), cudaMemcpyHostToDevice));

	backwardKernel<Batch, InFeatures, OutFeatures> << <numBlocks, threadsPerBlock >> > (d_weights, d_loss_gradient, d_output);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(output, d_output, Batch * InFeatures * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_weights);
	cudaFree(d_loss_gradient);
	cudaFree(d_output);
}

