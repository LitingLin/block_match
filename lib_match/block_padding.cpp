#include <string.h>
#include <type_traits>
#include "lib_match.h"

template <typename T>
inline T
abs_subtract(const T a, const T b, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr)
{
	if (a > b)
		return a - b;
	else
		return b - a;
}

template <typename T>
inline T
abs_subtract(const T a, const T b, typename std::enable_if<std::is_signed<T>::value>::type* = nullptr)
{
	return abs(a - b);
}

void determinePadSizeAccordingToPatchSize(int mat_M, int mat_N, int patch_M, int patch_N,
	int *M_left, int *M_right, int *N_left, int *N_right)
{
	int _M_left = mat_M % patch_M;
	int _N_left = mat_N % patch_N;
	int _M_right, _N_right;
	if (_M_left != 0)
		_M_right = mat_M - patch_M;
	else
		_M_right = 0;

	if (_N_left != 0)
		_N_right = mat_N - patch_N;
	else
		_N_right = 0;

	*M_left = _M_left;
	*M_right = _M_right;
	*N_left = _N_left;
	*N_right = _N_right;
}


/*******************
*      n           *
*    ----------    *
*  m | data -->    *
*    |             *
*    |             *
*                  *
*******************/
template <typename T>
void zeroPadding(const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post)
{
	size_t new_n = n + n_pre + n_post;
	size_t size = new_n * m_pre;
	memset(dst_ptr, 0, size * sizeof(T));
	dst_ptr += size;
	for (size_t i_row = 0; i_row != m; i_row++)
	{
		memset(dst_ptr, 0, n_pre * sizeof(T));
		dst_ptr += n_pre;
		memcpy(dst_ptr, src_ptr, n * sizeof(T));
		dst_ptr += n;
		src_ptr += n;
		memset(dst_ptr, 0, n_post * sizeof(T));
		dst_ptr += n_post;
	}
	size = new_n * m_post;
	memset(dst_ptr, 0, size * sizeof(T));
}

template <typename T>
void circularPadding(const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post)
{
	size_t new_n = n + n_pre + n_post;
	size_t new_m = m + m_pre + m_post;
	for (size_t i_y = 0; i_y != new_m; i_y++)
	{
		size_t old_i_y = abs_subtract(i_y, m_pre) % m;
		for (size_t i_x = 0; i_x != new_n; i_x++)
		{
			size_t old_i_x = abs_subtract(i_x, n_pre) % n;
			*dst_ptr = src_ptr[old_i_y * n + old_i_x];
			++dst_ptr;
		}
	}
}

template <typename T>
void replicatePadding(const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post)
{
	size_t new_n = n + n_pre + n_post;
	size_t new_m = m + m_pre + m_post;
	for (size_t i_y = 0; i_y != new_m; i_y++)
	{
		size_t old_i_y;
		if (i_y <= m_pre) old_i_y = 0;
		else if (i_y >= m_pre + m) old_i_y = m - 1;
		else old_i_y = i_y - m_pre;

		for (size_t i_x = 0; i_x != new_n; i_x++)
		{
			size_t old_i_x;
			if (i_x <= n_pre) old_i_x = 0;
			else if (i_x >= n_pre + n) old_i_x = n - 1;
			else old_i_x = i_x - n_pre;

			*dst_ptr = src_ptr[old_i_y * n + old_i_x];
			++dst_ptr;
		}
	}
}

// Ensure n_pre <= n, n_post <= n, m_pre <= m, m_post <= m
template <typename T>
void symmetricPadding(const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post)
{
	size_t new_n = n + n_pre + n_post;
	T *t1_ptr;
	T *t2_ptr;

	const T *t3_ptr;

	t3_ptr = src_ptr + n * (m_pre - 1);
	dst_ptr += n_pre;

	for (size_t i = 0; i != m_pre; i++)
	{
		t1_ptr = dst_ptr;

		memcpy(t1_ptr, t3_ptr, n * sizeof(T));

		t2_ptr = t1_ptr - 1;

		for (size_t j = 0; j != n_pre; j++)
			*t2_ptr-- = *t1_ptr++;

		t2_ptr = dst_ptr + n;
		t1_ptr = t2_ptr - 1;

		for (size_t j = 0; j != n_post; j++)
			*t2_ptr++ = *t1_ptr--;

		t3_ptr -= n;
		dst_ptr += new_n;
	}

	t3_ptr = src_ptr;
	for (size_t i = 0; i != m; i++)
	{
		t1_ptr = dst_ptr;
		memcpy(t1_ptr, t3_ptr, n * sizeof(T));
		t2_ptr = t1_ptr - 1;

		for (size_t j = 0; j != n_pre; j++)
			*t2_ptr-- = *t1_ptr++;
		t2_ptr = dst_ptr + n;
		t1_ptr = t2_ptr - 1;
		for (size_t j = 0; j != n_post; j++)
			*t2_ptr++ = *t1_ptr--;

		t3_ptr += n;
		dst_ptr += new_n;
	}

	t3_ptr = src_ptr + n * (m - 1);

	for (size_t i = 0; i != m_post; i++)
	{
		t1_ptr = dst_ptr;
		memcpy(t1_ptr, t3_ptr, n * sizeof(T));
		t2_ptr = t1_ptr - 1;

		for (size_t j = 0; j != n_pre; j++)
			*t2_ptr-- = *t1_ptr++;
		t2_ptr = dst_ptr + n;
		t1_ptr = t2_ptr - 1;
		for (size_t j = 0; j != n_post; j++)
			*t2_ptr++ = *t1_ptr--;

		t3_ptr -= n;
		dst_ptr += new_n;
	}
}

template
LIB_MATCH_EXPORT
void zeroPadding(const uint8_t *, uint8_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void zeroPadding(const int8_t *, int8_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void zeroPadding(const uint16_t *, uint16_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void zeroPadding(const int16_t *, int16_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void zeroPadding(const uint32_t *, uint32_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void zeroPadding(const int32_t *, int32_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void zeroPadding(const uint64_t *, uint64_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void zeroPadding(const int64_t *, int64_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void zeroPadding(const float *, float *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void zeroPadding(const double *, double *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const uint8_t *, uint8_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const int8_t *, int8_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const uint16_t *, uint16_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const int16_t *, int16_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const uint32_t *, uint32_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const int32_t *, int32_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const uint64_t *, uint64_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const int64_t *, int64_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const float *, float *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void circularPadding(const double *, double *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const uint8_t *, uint8_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const int8_t *, int8_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const uint16_t *, uint16_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const int16_t *, int16_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const uint32_t *, uint32_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const int32_t *, int32_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const uint64_t *, uint64_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const int64_t *, int64_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const float *, float *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void replicatePadding(const double *, double *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const uint8_t *, uint8_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const int8_t *, int8_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const uint16_t *, uint16_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const int16_t *, int16_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const uint32_t *, uint32_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const int32_t *, int32_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const uint64_t *, uint64_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const int64_t *, int64_t *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const float *, float *, size_t, size_t, size_t, size_t, size_t, size_t);
template
LIB_MATCH_EXPORT
void symmetricPadding(const double *, double *, size_t, size_t, size_t, size_t, size_t, size_t);
