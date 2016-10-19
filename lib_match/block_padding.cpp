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

template <typename T>
void zeroPadding(const T *old_ptr, T *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom)
{
	size_t new_width = old_width + pad_left + pad_right;
	size_t size;
	size = new_width * pad_up;
	memset(new_ptr, 0, size * sizeof(T));
	new_ptr += size;
	for (size_t i_row = 0; i_row != old_height; i_row++)
	{
		memset(new_ptr, 0, pad_left * sizeof(T));
		new_ptr += pad_left;
		memcpy(new_ptr, old_ptr, old_width * sizeof(T));
		new_ptr += old_width;
		old_ptr += old_width;
		memset(new_ptr, 0, pad_right * sizeof(T));
		new_ptr += pad_right;
	}
	size = new_width * pad_buttom;
	memset(new_ptr, 0, size * sizeof(T));
}


/*******************
*      x           *
*    ----------    *
*  y |             *
*    |             *
*    |             *
*                  *
*******************/

template <typename T>
void circularPadding(const T *old_ptr, T *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom)
{
	size_t new_width = old_width + pad_left + pad_right;
	size_t new_height = old_height + pad_up + pad_buttom;
	for (size_t i_y = 0; i_y != new_height; i_y++)
	{
		size_t old_i_y = abs_subtract(i_y, pad_up) % old_height;
		for (size_t i_x = 0; i_x != new_width; i_x++)
		{
			size_t old_i_x = abs_subtract(i_x, pad_left) % old_width;
			*new_ptr = old_ptr[old_i_y * old_width + old_i_x];
			++new_ptr;
		}
	}
}

template <typename T>
void replicatePadding(const T *old_ptr, T *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom)
{
	size_t new_width = old_width + pad_left + pad_right;
	size_t new_height = old_height + pad_up + pad_buttom;
	for (size_t i_y = 0; i_y != new_height; i_y++)
	{
		size_t old_i_y;
		if (i_y <= pad_up) old_i_y = 0;
		else if (i_y >= pad_up + old_height) old_i_y = old_height - 1;
		else old_i_y = i_y - pad_up;

		for (size_t i_x = 0; i_x != new_width; i_x++)
		{
			size_t old_i_x;
			if (i_x <= pad_left) old_i_x = 0;
			else if (i_x >= pad_left + old_width) old_i_x = old_width - 1;
			else old_i_x = i_x - pad_left;

			*new_ptr = old_ptr[old_i_y * old_width + old_i_x];
			++new_ptr;
		}
	}
}

// Ensure pad_left <= old_width, pad_right <= old_width, pad_up <= old_height, pad_buttom <= old_height
template <typename T>
void symmetricPadding(const T *old_ptr, T *new_ptr,
	size_t old_width, size_t old_height,
	size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom)
{
	size_t new_width = old_width + pad_left + pad_right;
	T *t1_ptr;
	T *t2_ptr;

	const T *t3_ptr;

	t3_ptr = old_ptr + old_width * (pad_up - 1);
	new_ptr += pad_left;

	for (size_t i = 0; i != pad_up; i++)
	{
		t1_ptr = new_ptr;

		memcpy(t1_ptr, t3_ptr, old_width * sizeof(T));

		t2_ptr = t1_ptr - 1;

		for (size_t j = 0; j != pad_left; j++)
			*t2_ptr-- = *t1_ptr++;

		t2_ptr = new_ptr + old_width;
		t1_ptr = t2_ptr - 1;

		for (size_t j = 0; j != pad_right; j++)
			*t2_ptr++ = *t1_ptr--;

		t3_ptr -= old_width;
		new_ptr += new_width;
	}

	t3_ptr = old_ptr;
	for (size_t i = 0; i != old_height; i++)
	{
		t1_ptr = new_ptr;
		memcpy(t1_ptr, t3_ptr, old_width * sizeof(T));
		t2_ptr = t1_ptr - 1;

		for (size_t j = 0; j != pad_left; j++)
			*t2_ptr-- = *t1_ptr++;
		t2_ptr = new_ptr + old_width;
		t1_ptr = t2_ptr - 1;
		for (size_t j = 0; j != pad_right; j++)
			*t2_ptr++ = *t1_ptr--;

		t3_ptr += old_width;
		new_ptr += new_width;
	}

	t3_ptr = old_ptr + old_width * (old_height - 1);

	for (size_t i = 0; i != pad_buttom; i++)
	{
		t1_ptr = new_ptr;
		memcpy(t1_ptr, t3_ptr, old_width * sizeof(T));
		t2_ptr = t1_ptr - 1;

		for (size_t j = 0; j != pad_left; j++)
			*t2_ptr-- = *t1_ptr++;
		t2_ptr = new_ptr + old_width;
		t1_ptr = t2_ptr - 1;
		for (size_t j = 0; j != pad_right; j++)
			*t2_ptr++ = *t1_ptr--;

		t3_ptr -= old_width;
		new_ptr += new_width;
	}

}
LIB_MATCH_EXPORT
template
void zeroPadding(const float *old_ptr, float *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
LIB_MATCH_EXPORT
template
void zeroPadding(const double *old_ptr, double *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
LIB_MATCH_EXPORT
template
void circularPadding(const float *old_ptr, float *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
LIB_MATCH_EXPORT
template
void circularPadding(const double *old_ptr, double *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
LIB_MATCH_EXPORT
template
void replicatePadding(const float *old_ptr, float *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
LIB_MATCH_EXPORT
template
void replicatePadding(const double *old_ptr, double *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
LIB_MATCH_EXPORT
template
void symmetricPadding(const float *old_ptr, float *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
LIB_MATCH_EXPORT
template
void symmetricPadding(const double *old_ptr, double *new_ptr, size_t old_width, size_t old_height, size_t pad_left, size_t pad_right, size_t pad_up, size_t pad_buttom);
