#pragma once

#include <stdbool.h>
#include <mex.h>

enum PadMethod
{
	Zero,
	Circular,
	Replicate,
	Symmetric
};

struct PaddingMexContext
{
	double *originImage;
	int image_M;
	int image_N;

	int pad_M_left;
	int pad_M_right;

	int pad_N_left;
	int pad_N_right;

	enum PadMethod method;
};

