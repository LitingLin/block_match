#include <cmath>

template <typename Type>
void standardize_cpu(Type *sequence, int size)
{
	Type mean = 0;
	for (int i=0;i<size;++i)
	{
		mean += sequence[i];
	}
	mean /= size;
	Type sd = 0;
	for (int i=0;i<size;++i)
	{
		Type t = sequence[i] -= mean;
		sd += t*t;
	}
	sd /= size;
	sd = sqrt(sd);
	for (int i=0;i<size;++i)
	{
		sequence[i] /= sd;
	}
}

template <typename Type>
void block_match_cc_cpu(Type *blocks_A, Type *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, Type *result)
{
	Type *c_blocks_A = blocks_A;
	Type *c_blocks_B = blocks_B;

	for (int index_A = 0; index_A < numberOfBlockA; ++index_A)
	{
		for (int index_B = 0; index_B < numberOfBlockBPerBlockA; ++index_B)
		{
			Type temp = 0;
			for (int index_in_block = 0; index_in_block < blockSize; ++index_in_block)
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
void standardize_cpu(float *sequence, int size);
template
void standardize_cpu(double *sequence, int size);
template
void block_match_cc_cpu(float *blocks_A, float *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, float *result);
template
void block_match_cc_cpu(double *blocks_A, double *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, double *result);