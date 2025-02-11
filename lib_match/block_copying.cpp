#include <memory>
#include <type_traits>
#include "template_instantiate_helper.h"

template <typename Type1, typename Type2,
	typename std::enable_if<std::is_same<Type1, Type2>::value>::type* = nullptr>
void copyBlock_helper(Type1 *buf, const Type2 *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N)
{
	Type1 *c_buf = buf;
	const Type2 *c_src = src + index_x * mat_N + index_y;
	for (int i = 0; i < block_M; ++i)
	{
		memcpy(c_buf, c_src, block_N * sizeof(Type1));
		c_buf += block_N;
		c_src += mat_N;
	}
}

#ifdef _MSC_VER
#pragma warning( disable : 4800 )  
#endif
template <typename Type1, typename Type2,
	typename std::enable_if<!std::is_same<Type1, Type2>::value>::type* = nullptr>
void copyBlock_helper(Type1 *buf, const Type2 *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N)
{
	Type1 *c_buf = buf;
	const Type2 *c_src = src + index_x * mat_N + index_y;
	for (int i = 0; i < block_M; ++i)
	{
		for (int j = 0; j < block_N; ++j)
			*c_buf++ = static_cast<Type1>(*c_src++);
		c_src += (mat_N - block_N);
	}
}
#ifdef _MSC_VER
#pragma warning( default : 4800 )  
#endif
// Workaround: MSVC SFINAE function pointer type cast bug
template <typename Type1, typename Type2>
void copyBlock(Type1 *buf, const Type2 *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N)
{
	copyBlock_helper(buf, src, mat_M, mat_N, index_x, index_y, block_M, block_N);
}

template <typename Type1, typename Type2>
void copyBlockMultiChannel(const size_t nChannels, Type1 *buf, const Type2 *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N)
{
	size_t blockSize = block_M * block_N;
	size_t matSize = mat_M * mat_N;
	for (size_t i=0;i<nChannels;++i)
	{
		copyBlock_helper(buf, src, mat_M, mat_N, index_x, index_y, block_M, block_N);
		buf += blockSize;
		src += matSize;
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
	if (index + block_length < 0) {
		index_pre_end = -(index + block_length);
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

template <typename Type>
void copyBlockWithSymmetricPadding(Type *buf, const Type *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N)
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
		const Type *c_mat = src + (i - 1) * mat_N;
		for (int j = y_index_pre_begin; j>y_index_pre_end; --j)
		{
			const Type *c_c_mat = c_mat + j - 1;
			*buf++ = *c_c_mat;
		}

		memcpy(buf, c_mat + y_index_begin, (y_index_end - y_index_begin) * sizeof(Type));
		buf += (y_index_end - y_index_begin);

		c_mat += mat_N - 1;
		for (int j = y_index_post_begin; j<y_index_post_end; ++j)
		{
			const Type *c_c_mat = c_mat - j;
			*buf++ = *c_c_mat;
		}
	}

	for (int i = x_index_begin; i<x_index_end; ++i)
	{
		const Type *c_mat = src + i * mat_N;
		for (int j = y_index_pre_begin; j>y_index_pre_end; --j)
		{
			const Type *c_c_mat = c_mat + j - 1;
			*buf++ = *c_c_mat;
		}

		memcpy(buf, c_mat + y_index_begin, (y_index_end - y_index_begin) * sizeof(Type));
		buf += (y_index_end - y_index_begin);

		c_mat += mat_N - 1;
		for (int j = y_index_post_begin; j<y_index_post_end; ++j)
		{
			const Type *c_c_mat = c_mat - j;
			*buf++ = *c_c_mat;
		}
	}

	for (int i = x_index_post_begin; i<x_index_post_end; ++i)
	{
		const Type *c_mat = src + (mat_M - i - 1) * mat_N;
		for (int j = y_index_pre_begin; j>y_index_pre_end; --j)
		{
			const Type *c_c_mat = c_mat + j - 1;
			*buf++ = *c_c_mat;
		}

		memcpy(buf, c_mat + y_index_begin, (y_index_end - y_index_begin) * sizeof(Type));
		buf += (y_index_end - y_index_begin);

		c_mat += mat_N - 1;
		for (int j = y_index_post_begin; j<y_index_post_end; ++j)
		{
			const Type *c_c_mat = c_mat - j;
			*buf++ = *c_c_mat;
		}
	}
}

#define exp(type1, type2) \
template \
void copyBlock(type1 *, const type2 *, int, int, int, int, int, int);
InstantiateTemplate2(exp);
#undef exp

#define exp(type1, type2) \
template \
void copyBlockMultiChannel(const size_t, type1 *, const type2 *, int, int, int, int, int, int);
InstantiateTemplate2(exp);
#undef exp

template
void copyBlockWithSymmetricPadding(float *buf, const float *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);
template
void copyBlockWithSymmetricPadding(double *buf, const double *src, int mat_M, int mat_N, int index_x, int index_y, int block_M, int block_N);