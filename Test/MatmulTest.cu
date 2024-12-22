#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <kernels.cuh>
#include <algorithm>

using ::testing::Pointwise;
using ::testing::FloatNear;

TEST(MatmulTest, MatmulTest)
{
	constexpr int N = 3;
	constexpr int M = 4;
	constexpr int P = 3;
	float h_A[N][M], h_B[M][P], h_C[N][P];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			h_A[i][j] = float(i * M + j);

	for (int i = 0; i < M; i++)
		for (int j = 0; j < P; j++)
			h_B[i][j] = float(i * P + j);

	matmul<N, M, P>(&h_A[0][0], &h_B[0][0], &h_C[0][0]);

	float expected[3][3] = { {42.f, 48.f, 54.f}, {114.f, 136.f, 158.f}, {186.f, 224.f, 262.f} };
	const float tolerance = 0.001f;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < P; j++)
			EXPECT_NEAR(expected[i][j], h_C[i][j], tolerance) << "Mismatch at index " << i << ', ' << j << std::endl;

}

TEST(ForwardTest, ForwardTest)
{
	constexpr int N = 3;
	constexpr int M = 4;
	float h_X[N], h_W[N][M], h_B[M], h_Y[M];

	for (int i = 0; i < N; i++)
		h_X[i] = float(i);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			h_W[i][j] = float(i * M + j);

	for (int i = 0; i < M; i++)
		h_B[i] = float(i);

	forward<N, M>(&h_X[0], &h_W[0][0], &h_B[0], &h_Y[0]);

	float expected[M] = {20.f, 24.f, 28.f, 32.f};
	const float tolerance = 0.001f;
	for (int i = 0; i < M; i++)
		EXPECT_NEAR(expected[i], h_Y[i], tolerance) << "Mismatch at index " << i << std::endl;
}

TEST(ReluTest, ReluTest) {
	constexpr int N = 10;
	constexpr int M = 1;
	float h_X[N], h_Y[N];

	for (int i = 0; i < N; i++)
		h_X[i] = float(i - 5);

	relu<N, M>(&h_X[0], &h_Y[0]);

	float expected[N] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 2.f, 3.f, 4.f};
	const float tolerance = 0.001f;
	for (int i = 0; i < N; i++)
		EXPECT_NEAR(expected[i], h_Y[i], tolerance) << "Mismatch at index " << i << std::endl;
}

TEST(SoftmaxTest, SoftmaxTest)
{
	constexpr int N = 2;
	constexpr int M = 3;
	float h_X[N][M] = {{4.f, 4.5f, -5.f}, {8.23f, 4.3345f, 5.232f}};
	float h_Y[N][M];

	softmax<N, M>(&h_X[0][0], &h_Y[0][0]);

	float expected[N][M] = { {0.3775230792f, 0.6224303308f, 0.00004659f}, {0.9343873681f, 0.0189990902f, 0.0466135417f} };
	const float tolerance = 0.0000001f;
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			EXPECT_NEAR(expected[i][j], h_Y[i][j], tolerance) << "Mismatch at index " << i << ', ' << j << std::endl;
}

TEST(CrossEntropyTest, CrossEntropyTest)
{
	constexpr int N = 2;
	constexpr int M = 3;
	float h_P[N][M] = { {0.7, 0.2, 0.1}, {0.1, 0.8, 0.1} };
	float h_Q[N][M] = { {1, 0, 0}, {0, 1, 0} };
	float h_H[N];

	crossEntropy<N, M>(&h_P[0][0], &h_Q[0][0], &h_H[0]);

	float expected[N] = { 0.35667494, 0.22314355 };
	const float tolerance = 0.0000001f;
	for (int i = 0; i < N; i++)
		EXPECT_NEAR(expected[i], h_H[i], tolerance) << "Mismatch at index " << i << std::endl;
}

TEST(InitRandomTest, InitRandomTest)
{
	constexpr int N = 2;
	constexpr int M = 3;
	float h_Y[N][M];

	initRandom<N, M>(&h_Y[0][0]);

	EXPECT_EQ(0, 0);
}