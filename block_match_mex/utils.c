#include "common.h"

void floatToDouble(const float *in, double *out, size_t n)
{
	for (size_t i = 0; i != n; i++)
	{
		out[i] = in[i];
	}
}

void doubleToFloat(const double *in, float *out, size_t n)
{
	for (size_t i = 0; i != n; i++)
	{
		out[i] = in[i];
	}
}