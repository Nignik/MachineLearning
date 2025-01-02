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

TEST(CrossEntropyBackwardsTest, CrossEntropyBackwardsTest)
{
	constexpr int classes = 3;
	constexpr int batches = 4;
	float gradient[batches][classes];

	float predictions[batches][classes] = {
		{0.406902, 0.264210, 0.328889},
		{0.272696, 0.440727, 0.286577},
		{0.189005, 0.388212, 0.422783},
		{0.311934, 0.371288, 0.316778}
	};
	float labels[batches][classes] = {
		{0.000000, 1.000000, 0.000000},
		{1.000000, 0.000000, 0.000000},
		{0.000000, 1.000000, 0.000000},
		{0.000000, 0.000000, 1.000000}
	};

	float expected_gradient[batches][classes] = {
		{0.406902, -0.735790, 0.328889},
		{-0.727304, 0.440727, 0.286577},
		{0.189005, -0.611788, 0.422783},
		{0.311934, 0.371288, -0.683222}
	};

	crossEntropyBackwards<batches, classes>(&predictions[0][0], &labels[0][0], &gradient[0][0]);

	const float tolerance = 0.0000001f;
	for (int i = 0; i < batches; i++)
		for (int j = 0; j < classes; j++)
			EXPECT_NEAR(expected_gradient[i][j], gradient[i][j], tolerance) << "Mismatch at index " << i << ', ' << j << std::endl;
}

TEST(BackwardTest, BackwardTest)
{
	constexpr int inFeatures = 3;
	constexpr int outFeatures = 4;
	constexpr int batches = 4;

	float weights[inFeatures][outFeatures] = {
		{0.374540, 0.950714, 0.731994, 0.598658},
		{0.156019, 0.155995, 0.058084, 0.866176},
		{0.601115, 0.708073, 0.020584, 0.969910}
	};

	float gradientNext[batches][outFeatures] = {
		{0.832443, 0.212339, 0.181825, 0.183405},
		{0.304242, 0.524756, 0.431945, 0.291229},
		{0.611853, 0.139494, 0.292145, 0.366362},
		{0.456070, 0.785176, 0.199674, 0.514234}
	};

	float expectedOutput[batches][inFeatures] = {
		{0.756548, 0.332422, 0.832374},
		{1.103372, 0.406671, 0.845808},
		{0.794956, 0.451523, 0.827917},
		{1.371305, 0.650654, 1.332983}
	};

	float output[batches][inFeatures];
	backward<batches, inFeatures, outFeatures>(&weights[0][0], &gradientNext[0][0], &output[0][0]);

	const float tolerance = 0.000001f;
	for (int i = 0; i < batches; i++)
		for (int j = 0; j < inFeatures; j++)
			EXPECT_NEAR(expectedOutput[i][j], output[i][j], tolerance) << "Mismatch at index " << i << ', ' << j << std::endl;
}