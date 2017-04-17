#include "intrinsic_func_helper.hpp"

#include <cmath>

float accumulate_avx(float *begin, float *end)
{
	__m256 t = _mm256_setzero_ps();
	for (float * iter = begin; iter < end; iter += sizeof(__m256) / sizeof(float))
	{
		__m256 a = _mm256_load_ps(iter);
		t = _mm256_add_ps(a, t);
	}
	return sum8f_avx(t);
}

double accumulate_avx(double *begin, double *end)
{
	__m256d t = _mm256_setzero_pd();
	for (double * iter = begin; iter < end; iter += sizeof(__m256d) / sizeof(double))
	{
		__m256d a = _mm256_load_pd(iter);
		t = _mm256_add_pd(a, t);
	}
	return sum4d_avx(t);
}

template <typename Type>
void lib_match_cc_cpu_avx2(Type *, Type *, int, Type *)
{
	static_assert("NOT IMPLEMENTED");
}

template <>
void lib_match_cc_cpu_avx2(float *block_A, float *block_B, int blockSize, float *result)
{
	__m256 X, Y, Z;
	X = Y = Z = _mm256_setzero_ps();

	__m256 A_mean = _mm256_set1_ps(accumulate_avx(block_A, block_A + blockSize));
	__m256 B_mean = _mm256_set1_ps(accumulate_avx(block_B, block_B + blockSize));

	for (int i = 0; i < blockSize; i += sizeof(__m256) / sizeof(float))
	{
		__m256 M = _mm256_sub_ps(_mm256_load_ps(block_A + i), A_mean);
		__m256 N = _mm256_sub_ps(_mm256_load_ps(block_B + i), B_mean);

		X = _mm256_fmadd_ps(M, N, X);
		Y = _mm256_fmadd_ps(M, M, Y);
		Z = _mm256_fmadd_ps(N, N, Z);
	}

	float X_scalar = sum8f_avx(X), Y_scalar = sum8f_avx(Y), Z_scalar = sum8f_avx(Z);
	*result = X_scalar / std::sqrt(Y_scalar * Z_scalar);
}

template <>
void lib_match_cc_cpu_avx2(double *block_A, double *block_B, int blockSize, double *result)
{
	__m256d X, Y, Z;
	X = Y = Z = _mm256_setzero_pd();

	__m256d A_mean = _mm256_set1_pd(accumulate_avx(block_A, block_A + blockSize));
	__m256d B_mean = _mm256_set1_pd(accumulate_avx(block_B, block_B + blockSize));

	for (int i = 0; i < blockSize; i += sizeof(__m256d) / sizeof(double))
	{
		__m256d M = _mm256_sub_pd(_mm256_load_pd(block_A + i), A_mean);
		__m256d N = _mm256_sub_pd(_mm256_load_pd(block_B + i), B_mean);

		X = _mm256_fmadd_pd(M, N, X);
		Y = _mm256_fmadd_pd(M, M, Y);
		Z = _mm256_fmadd_pd(N, N, Z);
	}

	double X_scalar = sum4d_avx(X), Y_scalar = sum4d_avx(Y), Z_scalar = sum4d_avx(Z);
	*result = X_scalar / std::sqrt(Y_scalar * Z_scalar);
}

template <typename Type>
void lib_match_cc_cpu_avx(Type *, Type *, int, Type *)
{
	static_assert("NOT IMPLEMENTED");
}

template <>
void lib_match_cc_cpu_avx(float *block_A, float *block_B, int blockSize, float *result)
{
	__m256 X, Y, Z;
	X = Y = Z = _mm256_setzero_ps();

	__m256 A_mean = _mm256_set1_ps(accumulate_avx(block_A, block_A + blockSize));
	__m256 B_mean = _mm256_set1_ps(accumulate_avx(block_B, block_B + blockSize));

	for (int i = 0; i < blockSize; i += sizeof(__m256) / sizeof(float))
	{
		__m256 M = _mm256_sub_ps(_mm256_load_ps(block_A + i), A_mean);
		__m256 N = _mm256_sub_ps(_mm256_load_ps(block_B + i), B_mean);

		X = _mm256_add_ps(_mm256_mul_ps(M, N), X);
		Y = _mm256_add_ps(_mm256_mul_ps(M, M), Y);
		Z = _mm256_add_ps(_mm256_mul_ps(N, N), Z);
	}

	float X_scalar = sum8f_avx(X), Y_scalar = sum8f_avx(Y), Z_scalar = sum8f_avx(Z);
	*result = X_scalar / std::sqrt(Y_scalar * Z_scalar);
}

template <>
void lib_match_cc_cpu_avx(double *block_A, double *block_B, int blockSize, double *result)
{
	__m256d X, Y, Z;
	X = Y = Z = _mm256_setzero_pd();

	__m256d A_mean = _mm256_set1_pd(accumulate_avx(block_A, block_A + blockSize));
	__m256d B_mean = _mm256_set1_pd(accumulate_avx(block_B, block_B + blockSize));

	for (int i = 0; i < blockSize; i += sizeof(__m256d) / sizeof(double))
	{
		__m256d M = _mm256_sub_pd(_mm256_load_pd(block_A + i), A_mean);
		__m256d N = _mm256_sub_pd(_mm256_load_pd(block_B + i), B_mean);

		X = _mm256_add_pd(_mm256_mul_pd(M, N), X);
		Y = _mm256_add_pd(_mm256_mul_pd(M, M), Y);
		Z = _mm256_add_pd(_mm256_mul_pd(N, N), Z);
	}

	double X_scalar = sum4d_avx(X), Y_scalar = sum4d_avx(Y), Z_scalar = sum4d_avx(Z);
	*result = X_scalar / std::sqrt(Y_scalar * Z_scalar);
}

float accumulate_sse3(float *begin, float *end)
{
	__m128 t = _mm_setzero_ps();
	for (float * iter = begin; iter < end; iter += sizeof(__m128) / sizeof(float))
	{
		__m128 a = _mm_load_ps(iter);
		t = _mm_add_ps(a, t);
	}
	return sum4f_sse3(t);
}

double accumulate_sse3(double *begin, double *end)
{
	__m128d t = _mm_setzero_pd();
	for (double * iter = begin; iter < end; iter += sizeof(__m128d) / sizeof(double))
	{
		__m128d a = _mm_load_pd(iter);
		t = _mm_add_pd(a, t);
	}
	return sum2d_sse3(t);
}

template <typename Type>
void lib_match_cc_cpu_sse3(Type *, Type *, int, Type *)
{
	static_assert("NOT IMPLEMENTED");
}

template <>
void lib_match_cc_cpu_sse3(float *block_A, float *block_B, int blockSize, float *result)
{
	__m128 X, Y, Z;
	X = Y = Z = _mm_setzero_ps();

	__m128 A_mean = _mm_set1_ps(accumulate_sse3(block_A, block_A + blockSize));
	__m128 B_mean = _mm_set1_ps(accumulate_sse3(block_B, block_B + blockSize));

	for (int i = 0; i < blockSize; i += sizeof(__m128) / sizeof(float))
	{
		__m128 M = _mm_sub_ps(_mm_load_ps(block_A + i), A_mean);
		__m128 N = _mm_sub_ps(_mm_load_ps(block_B + i), B_mean);

		X = _mm_add_ps(_mm_mul_ps(M, N), X);
		Y = _mm_add_ps(_mm_mul_ps(M, M), Y);
		Z = _mm_add_ps(_mm_mul_ps(N, N), Z);
	}

	float X_scalar = sum4f_sse3(X), Y_scalar = sum4f_sse3(Y), Z_scalar = sum4f_sse3(Z);
	*result = X_scalar / std::sqrt(Y_scalar * Z_scalar);
}

template <>
void lib_match_cc_cpu_sse3(double *block_A, double *block_B, int blockSize, double *result)
{
	__m128d X, Y, Z;
	X = Y = Z = _mm_setzero_pd();

	__m128d A_mean = _mm_set1_pd(accumulate_sse3(block_A, block_A + blockSize));
	__m128d B_mean = _mm_set1_pd(accumulate_sse3(block_B, block_B + blockSize));

	for (int i = 0; i < blockSize; i += sizeof(__m128d) / sizeof(double))
	{
		__m128d M = _mm_sub_pd(_mm_load_pd(block_A + i), A_mean);
		__m128d N = _mm_sub_pd(_mm_load_pd(block_B + i), B_mean);

		X = _mm_add_pd(_mm_mul_pd(M, N), X);
		Y = _mm_add_pd(_mm_mul_pd(M, M), Y);
		Z = _mm_add_pd(_mm_mul_pd(N, N), Z);
	}

	double X_scalar = sum2d_sse3(X), Y_scalar = sum2d_sse3(Y), Z_scalar = sum2d_sse3(Z);
	*result = X_scalar / std::sqrt(Y_scalar * Z_scalar);
}

#include <numeric>

template <typename Type>
void lib_match_cc_cpu(Type *block_A, Type *block_B, int blockSize, Type *result)
{
	Type X = 0, Y = 0, Z = 0;
	Type A_mean = std::accumulate(block_A, block_A + blockSize, (Type)0) / (Type)blockSize;
	Type B_mean = std::accumulate(block_B, block_B + blockSize, (Type)0) / (Type)blockSize;

	for (int i = 0; i < blockSize; ++i)
	{
		Type M = block_A[i] - A_mean;
		Type N = block_B[i] - B_mean;
		X += M*N;
		Y += M*M;
		Z += N*N;
	}

	*result = X / std::sqrt(Y*Z);
}

template
void lib_match_cc_cpu(float *, float *, int, float *);
template
void lib_match_cc_cpu(double *, double *, int, double *);