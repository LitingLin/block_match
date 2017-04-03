#define __AVX2__

#ifdef __AVX2__
#include <immintrin.h>

template <typename Type>
void block_match_mse_cpu(Type *, Type *, int, Type *)
{
	static_assert("NOT IMPLEMENTED");
}

template <>
void block_match_mse_cpu(float *block_A, float *block_B, int blockSize, float *result)
{
	__m256 t = _mm256_setzero_ps();
	for (int i = 0; i < blockSize; i += sizeof(__m256) / sizeof(float))
	{
		__m256 a = _mm256_load_ps(block_A);
		__m256 b = _mm256_load_ps(block_B);
		a = _mm256_sub_ps(a, b);
		t = _mm256_fmadd_ps(a, a, t);
	}
	t = _mm256_hadd_ps(t, t);
	t = _mm256_hadd_ps(t, t);
	t = _mm256_hadd_ps(t, t);
	*result = t.m256_f32[0] / blockSize;
}

template <>
void block_match_mse_cpu(double *block_A, double *block_B, int blockSize, double *result)
{
	__m256d t = _mm256_setzero_pd();
	for (int i = 0; i < blockSize; i += sizeof(__m256d) / sizeof(double))
	{
		__m256d a = _mm256_load_pd(block_A);
		__m256d b = _mm256_load_pd(block_B);
		a = _mm256_sub_pd(a, b);
		t = _mm256_fmadd_pd(a, a, t);
	}
	t = _mm256_hadd_pd(t, t);
	t = _mm256_hadd_pd(t, t);
	*result = t.m256d_f64[0] / blockSize;
}

#elif defined __AVX__

#include <immintrin.h>

template <typename Type>
void block_match_mse_cpu(Type *, Type *, int, Type *)
{
	static_assert("NOT IMPLEMENTED");
}

template <>
void block_match_mse_cpu(float *block_A, float *block_B, int blockSize, float *result)
{
	__m256 t = _mm256_setzero_ps();
	for (int i = 0; i < blockSize; i += sizeof(__m256) / sizeof(float))
	{
		__m256 a = _mm256_load_ps(block_A);
		__m256 b = _mm256_load_ps(block_B);
		a = _mm256_sub_ps(a, b);
		a = _mm256_mul_ps(a, a);
		t = _mm256_add_ps(a, t);
	}
	t = _mm256_hadd_ps(t, t);
	t = _mm256_hadd_ps(t, t);
	t = _mm256_hadd_ps(t, t);
	*result = t.m256_f32[0] / blockSize;
}

template <>
void block_match_mse_cpu(double *block_A, double *block_B, int blockSize, double *result)
{
	__m256d t = _mm256_setzero_pd();
	for (int i = 0; i < blockSize; i += sizeof(__m256d) / sizeof(double))
	{
		__m256d a = _mm256_load_pd(block_A);
		__m256d b = _mm256_load_pd(block_B);
		a = _mm256_sub_pd(a, b);
		a = _mm256_mul_pd(a, a);
		t = _mm256_add_pd(a, t);
	}
	t = _mm256_hadd_pd(t, t);
	t = _mm256_hadd_pd(t, t);
	*result = t.m256d_f64[0] / blockSize;
}

#elif defined __SSE3__

#include <pmmintrin.h>

template <typename Type>
void block_match_mse_cpu(Type *block_A, Type *block_B, int blockSize, Type *result)
{
	static_assert("NOT IMPLEMENTED");
}

template <>
void block_match_mse_cpu(float *block_A, float *block_B, int blockSize, float *result)
{
	__m128 t = _mm_setzero_ps();
	for (int i = 0; i < blockSize; i += sizeof(__m128) / sizeof(float))
	{
		__m128 a = _mm_load_ps(block_A);
		__m128 b = _mm_load_ps(block_B);
		a = _mm_sub_ps(a, b);
		a = _mm_mul_ps(a, a);
		t = _mm_add_ps(a, t);
	}
	t = _mm_hadd_ps(t, t);
	t = _mm_hadd_ps(t, t);
	*result = t.m128_f32[0] / blockSize;
}

template <>
void block_match_mse_cpu(double *block_A, double *block_B, int blockSize, double *result)
{
	__m128d t = _mm_setzero_pd();
	for (int i = 0; i < blockSize; i += sizeof(__m128d) / sizeof(double))
	{
		__m128d a = _mm_load_pd(block_A);
		__m128d b = _mm_load_pd(block_B);
		a = _mm_sub_pd(a, b);
		a = _mm_mul_pd(a, a);
		t = _mm_add_pd(a, t);
	}
	t = _mm_hadd_pd(t, t);
	*result = t.m128d_f64[0] / blockSize;
}

#else
template <typename Type>
void block_match_mse_cpu(Type *blocks_A, Type *blocks_B, int blockSize, Type *result)
{
	Type temp = 0;
	for (int index_in_block = 0; index_in_block < blockSize; ++index_in_block)
	{
		Type v = blocks_A[index_in_block] - blocks_B[index_in_block];
		temp += v*v;
	}

	*result = temp / blockSize;
}

template
void block_match_mse_cpu(float *blocks_A, float *blocks_B, int blockSize, float *result);
template
void block_match_mse_cpu(double *blocks_A, double *blocks_B, int blockSize, double *result);

#endif