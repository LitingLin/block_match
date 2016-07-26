#include "block_match_internal.h"

void copyBlock(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N)
{
	float *c_buf = buf;
	const float *c_src = src + index_x * mat_M + index_y;
	for (int i = 0; i < block_N; ++i)
	{
		memcpy(c_buf, c_src, block_M * sizeof(float));
		c_buf += block_M;
		c_src += mat_M;
	}
}

void determineIndexPreMat(int index, int mat_length, int block_length, int &index_pre_begin, int &index_pre_end)
{
	if (index >= 0) {
		index_pre_begin = 0;
		index_pre_end = 0;
		return;
	}

	index_pre_begin = -index;
	if (index + (int)block_length < 0) {
		index_pre_end = -(index + (int)block_length);
	}
	else {
		index_pre_end = 0;
	}
}

void determineIndexInMat(int index, int mat_length, int block_length, int &index_begin, int &index_end)
{
	if (index >= mat_length)
	{
		index_begin = 0;
		index_end = 0;
		return;
	}

	if (index < 0)
		index_begin = 0;
	else
		index_begin = index;

	if (index + block_length < 0)
	{
		index_end = 0;
	}
	else if (index + block_length < mat_length)
	{
		index_end = index + block_length;
	}
	else
	{
		index_end = mat_length;
	}
}

void determinIndexPostMat(int index, int mat_length, int block_length, int &index_post_begin, int &index_post_end)
{
	if (index < mat_length)
		index_post_begin = 0;
	else
		index_post_begin = index - mat_length;

	if (index + block_length < mat_length)
		index_post_end = 0;
	else
		index_post_end = index + block_length - mat_length;
}

void determineIndex(int index, int mat_length, int block_length, 
	int &index_pre_begin, int &index_pre_end,
	int &index_begin, int &index_end, 
	int &index_post_begin, int &index_post_end)
{
	determineIndexPreMat(index, mat_length, block_length, index_pre_begin, index_pre_end);
	determineIndexInMat(index, mat_length, block_length, index_begin, index_end);
	determinIndexPostMat(index, mat_length, block_length, index_post_begin, index_post_end);
}

void copyBlockWithSymmetricPaddding(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N)
{
	if (index_x >= 0 && index_y >= 0 && index_x + block_M < mat_M && index_y + block_N < mat_N)
	{
		copyBlock(buf, src, mat_M, mat_N, index_x, index_y, block_M, block_N);
		return;
	}

	int x_index_pre_begin, x_index_pre_end, x_index_begin, x_index_end, x_index_post_begin, x_index_post_end;
	int y_index_pre_begin, y_index_pre_end, y_index_begin, y_index_end, y_index_post_begin, y_index_post_end;

	determineIndex(index_x, mat_M, block_M, x_index_pre_begin, x_index_pre_end, x_index_begin, x_index_end, x_index_post_begin, x_index_post_end);
	determineIndex(index_y, mat_N, block_N, y_index_pre_begin, y_index_pre_end, y_index_begin, y_index_end, y_index_post_begin, y_index_post_end);

	for (int i = x_index_pre_begin; i>x_index_pre_end; --i)
	{
		const float *c_mat = src + (i - 1) * mat_M;
		for (int j = y_index_pre_begin; j>y_index_pre_end; --j)
		{
			const float *c_c_mat = c_mat + j - 1;
			*buf++ = *c_c_mat;
		}

		memcpy(buf, c_mat + y_index_begin, (y_index_end - y_index_begin) * sizeof(float));
		buf += (y_index_end - y_index_begin);

		c_mat += mat_M - 1;
		for (int j = y_index_post_begin; j<y_index_post_end; ++j)
		{
			const float *c_c_mat = c_mat - j;
			*buf++ = *c_c_mat;
		}
	}

	for (int i = x_index_begin; i<x_index_end; ++i)
	{
		const float *c_mat = src + i * mat_M;
		for (int j = y_index_pre_begin; j>y_index_pre_end; --j)
		{
			const float *c_c_mat = c_mat + j - 1;
			*buf++ = *c_c_mat;
		}

		memcpy(buf, c_mat + y_index_begin, (y_index_end - y_index_begin) * sizeof(float));
		buf += (y_index_end - y_index_begin);

		c_mat += mat_M - 1;
		for (int j = y_index_post_begin; j<y_index_post_end; ++j)
		{
			const float *c_c_mat = c_mat - j;
			*buf++ = *c_c_mat;
		}
	}

	for (int i = x_index_post_begin; i<x_index_post_end; ++i)
	{
		const float *c_mat = src + (mat_N - i - 1) * mat_M;
		for (int j = y_index_pre_begin; j>y_index_pre_end; --j)
		{
			const float *c_c_mat = c_mat + j - 1;
			*buf++ = *c_c_mat;
		}

		memcpy(buf, c_mat + y_index_begin, (y_index_end - y_index_begin) * sizeof(float));
		buf += (y_index_end - y_index_begin);

		c_mat += mat_M - 1;
		for (int j = y_index_post_begin; j<y_index_post_end; ++j)
		{
			const float *c_c_mat = c_mat - j;
			*buf++ = *c_c_mat;
		}
	}
}