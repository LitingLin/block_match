#include <string.h>
#include <type_traits>
#include "lib_match.h"
#include "template_instantiate_helper.h"

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


template <typename T>
void zeroPaddingMultiChannel(const size_t nChannels, const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post)
{
	size_t newM = m + m_pre + m_post;
	size_t newN = n + n_pre + n_post;

	for (size_t i = 0; i < nChannels; ++i) {
		zeroPadding(src_ptr, dst_ptr, m, n, m_pre, m_post, n_pre, n_post);
		src_ptr += m * n;
		dst_ptr += newM * newN;
	}
}

template <typename T>
void circularPaddingMultiChannel(const size_t nChannels, const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post)
{
	size_t newM = m + m_pre + m_post;
	size_t newN = n + n_pre + n_post;

	for (size_t i = 0; i < nChannels; ++i) {
		circularPadding(src_ptr, dst_ptr, m, n, m_pre, m_post, n_pre, n_post);
		src_ptr += m * n;
		dst_ptr += newM * newN;
	}	
}

template <typename T>
void replicatePaddingMultiChannel(const size_t nChannels, const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post)
{
	size_t newM = m + m_pre + m_post;
	size_t newN = n + n_pre + n_post;

	for (size_t i = 0; i < nChannels; ++i) {
		replicatePadding(src_ptr, dst_ptr, m, n, m_pre, m_post, n_pre, n_post);
		src_ptr += m * n;
		dst_ptr += newM * newN;
	}
}

template <typename T>
void symmetricPaddingMultiChannel(const size_t nChannels, const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post)
{
	size_t newM = m + m_pre + m_post;
	size_t newN = n + n_pre + n_post;

	for (size_t i = 0; i < nChannels; ++i) {
		symmetricPadding(src_ptr, dst_ptr, m, n, m_pre, m_post, n_pre, n_post);
		src_ptr += m * n;
		dst_ptr += newM * newN;
	}
}

#define EXP(type) \
template \
LIB_MATCH_EXPORT \
void zeroPadding(const type *, type *, size_t, size_t, size_t, size_t, size_t, size_t)
InstantiateTemplate(EXP);
#undef EXP

#define EXP(type) \
template \
LIB_MATCH_EXPORT \
void circularPadding(const type *, type *, size_t, size_t, size_t, size_t, size_t, size_t)
InstantiateTemplate(EXP);
#undef EXP

#define EXP(type) \
template \
LIB_MATCH_EXPORT \
void replicatePadding(const type *, type *, size_t, size_t, size_t, size_t, size_t, size_t)
InstantiateTemplate(EXP);
#undef EXP

#define EXP(type) \
template \
LIB_MATCH_EXPORT \
void symmetricPadding(const type *, type *, size_t, size_t, size_t, size_t, size_t, size_t)
InstantiateTemplate(EXP);
#undef EXP

#define EXP(type) \
template \
LIB_MATCH_EXPORT \
void zeroPaddingMultiChannel(const size_t, const type *, type *, size_t, size_t, size_t, size_t, size_t, size_t)
InstantiateTemplate(EXP);
#undef EXP

#define EXP(type) \
template \
LIB_MATCH_EXPORT \
void circularPaddingMultiChannel(const size_t, const type *, type *, size_t, size_t, size_t, size_t, size_t, size_t)
InstantiateTemplate(EXP);
#undef EXP

#define EXP(type) \
template \
LIB_MATCH_EXPORT \
void replicatePaddingMultiChannel(const size_t, const type *, type *, size_t, size_t, size_t, size_t, size_t, size_t)
InstantiateTemplate(EXP);
#undef EXP

#define EXP(type) \
template \
LIB_MATCH_EXPORT \
void symmetricPaddingMultiChannel(const size_t, const type *, type *, size_t, size_t, size_t, size_t, size_t, size_t)
InstantiateTemplate(EXP);
#undef EXP