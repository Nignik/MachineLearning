#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <kernels.cuh>

using ::testing::Pointwise;
using ::testing::FloatNear;

TEST(MatmulTest, Matmul)
{
	constexpr int N = 3;
	constexpr int M = 4;
	constexpr int P = 3;
	float h_A[N][M], h_B[M][P], h_C[N][P];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			h_A[i][j] = float(i * M + j);
		}
	}

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < P; j++)
		{
			h_B[i][j] = float(i * P + j);
		}
	}

	matmul<N, M, P>(&h_A[0][0], &h_B[0][0], &h_C[0][0]);

	float expected[3][3] = { {42.f, 48.f, 54.f}, {114.f, 136.f, 158.f}, {186.f, 224.f, 262.f} };
	const float tolerance = 0.001f;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < P; j++)
		{
			EXPECT_NEAR(expected[i][j], h_C[i][j], tolerance) << "Mismatch at index " << i << ', ' << j;
		}
	}
}