void block_match_mse_cpu(float *blocks_A, float *blocks_B, int numberOfBlockA, int numberOfBlockBPerBlockA, int blockSize, float *result)
{
	float *c_blocks_A = blocks_A;
	float *c_blocks_B = blocks_B;

	for (int index_A = 0; index_A < numberOfBlockA; ++index_A) 
	{
		for (int index_B = 0; index_B < numberOfBlockBPerBlockA; ++index_B)
		{
			float temp = 0;
			for (int index_in_block = 0; index_in_block < blockSize;++index_in_block)
			{
				float v = c_blocks_A[index_in_block] - c_blocks_B[index_in_block];
				temp += v*v;
			}
			*result++ = temp / blockSize;

			blocks_B += blockSize;
		}
		blocks_A += blockSize;
	}
}