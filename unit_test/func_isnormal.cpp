#include "test_common.h"

template <typename Type>
void checkIsNormal(Type *ptr, int size)
{
	Type mean = 0;
	for (int i = 0; i<size; ++i)
	{
		mean += ptr[i];
	}
	mean /= size;

	BOOST_CHECK_SMALL(mean, (Type)singleFloatingPointErrorTolerance);

	Type sd = 0;

	for (int i = 0; i<size; ++i)
	{
		sd += ptr[i] * ptr[i];
	}
	sd /= size;
	sd = sqrt(sd);
	BOOST_CHECK_SMALL(sd - 1, (Type)singleFloatingPointErrorTolerance);
}

template
void checkIsNormal(float *ptr, int size);
template
void checkIsNormal(double *ptr, int size);