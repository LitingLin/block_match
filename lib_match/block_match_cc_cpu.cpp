#include <immintrin.h>

__m256 accumulate_avx(float *begin, float *end)
{
	__m256 t = _mm256_setzero_ps();
	for (float * iter = begin; iter < end; iter += sizeof(__m256) / sizeof(float))
	{
		__m256 a = _mm256_load_ps(iter);
		t = _mm256_add_ps(a, t);
	}
	t = _mm256_hadd_ps(t, t);
	t = _mm256_hadd_ps(t, t);
	t = _mm256_hadd_ps(t, t);
	return t;
}

__m256d accumulate_avx(double *begin, double *end)
{
	__m256d t = _mm256_setzero_pd();
	for (double * iter = begin; iter < end; iter += sizeof(__m256d) / sizeof(double))
	{
		__m256d a = _mm256_load_pd(iter);
		t = _mm256_add_pd(a, t);
	}
	t = _mm256_hadd_pd(t, t);
	t = _mm256_hadd_pd(t, t);
	return t;
}

template <typename Type>
void lib_match_cc_cpu_avx2(Type *block_A, Type *block_B, int blockSize, Type *result)
{
	
}

#include <cmath>
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