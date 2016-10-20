#pragma once

#include <lib_match_mex_common.h>

struct PaddingMexContext
{
	double *originImage;
	int image_M;
	int image_N;

	int pad_M_pre;
	int pad_M_post;

	int pad_N_pre;
	int pad_N_post;

	PadMethod method;
};

struct LibMatchMexErrorWithMessage parseParameter(struct PaddingMexContext *context,
	int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]);