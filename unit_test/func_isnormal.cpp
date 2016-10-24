#include "test_common.h"

template <typename Type>
void isNormal(Type *ptr, int size)
{
	float mean = 0;
	for (int i = 0; i<size; ++i)
	{
		mean += ptr[i];
	}
	mean /= size;

	BOOST_CHECK_SMALL(mean, 0.0001f);

	float sd = 0;

	for (int i = 0; i<size; ++i)
	{
		sd += ptr[i] * ptr[i];
	}
	sd /= size;
	sd = sqrt(sd);
	BOOST_CHECK_SMALL(sd - 1, 0.0001f);
}

template
void isNormal(float *ptr, int size);
template
void isNormal(double *ptr, int size);