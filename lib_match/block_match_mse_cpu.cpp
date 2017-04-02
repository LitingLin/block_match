#define __AVX2__

#ifdef __AVX2__
#include <immintrin.h>

template <typename Type>
void block_match_mse_cpu(Type *blocks_A, Type *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, Type *result)
{
	static_assert("NOT IMPLEMENTED");
}

template <>
void block_match_mse_cpu(float *blocks_A, float *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, float *result)
{
	
}

template <>
void block_match_mse_cpu(double *blocks_A, double *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, double *result)
{

}

#elif defined __AVX__

template <typename Type>
void block_match_mse_cpu(Type *blocks_A, Type *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, Type *result)
{
	static_assert("NOT IMPLEMENTED");
}

template <>
void block_match_mse_cpu(float *blocks_A, float *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, float *result)
{

}

template <>
void block_match_mse_cpu(double *blocks_A, double *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, double *result)
{

}

#elif defined __SSE2__

template <typename Type>
void block_match_mse_cpu(Type *blocks_A, Type *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, Type *result)
{
	static_assert("NOT IMPLEMENTED");
}

template <>
void block_match_mse_cpu(float *blocks_A, float *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, float *result)
{

}

template <>
void block_match_mse_cpu(double *blocks_A, double *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, double *result)
{

}

#else
template <typename Type>
void block_match_mse_cpu(Type *blocks_A, Type *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, Type *result)
{
	Type *c_blocks_A = blocks_A;
	Type *c_blocks_B = blocks_B;

	for (int index_A = 0; index_A < numberOfBlockA; ++index_A) 
	{
		for (int index_B = 0; index_B < numberOfBlockBPerBlockA; ++index_B)
		{
			Type temp = 0;
			for (int index_in_block = 0; index_in_block < blockSize;++index_in_block)
			{
				Type v = c_blocks_A[index_in_block] - c_blocks_B[index_in_block];
				temp += v*v;
			}
			*result++ = temp / blockSize;

			blocks_B += blockSize;
		}
		blocks_A += blockSize;
	}
}

template
void block_match_mse_cpu(float *blocks_A, float *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, float *result);
template
void block_match_mse_cpu(double *blocks_A, double *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, double *result);

#endif