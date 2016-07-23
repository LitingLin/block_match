#include "common.h"

#include "utils.h"

bool generate_result(mxArray **_pa, const int sequenceAHeight, const int sequenceAWidth, const int *index, const float *value, const int size)
{
	const int *c_index = index;
	const float *c_value = value;

	mxArray *pa = mxCreateCellMatrix(sequenceAHeight, sequenceAWidth);
	for (int i=0;i<sequenceAHeight;i++)
		for (int j = 0; j < sequenceAWidth; j++)
		{
			mxArray *mat = mxCreateNumericMatrix(size, 2, mxDOUBLE_CLASS, mxREAL);
			if (!mat)
				return false;

			double *mat_ptr = mxGetPr(mat);
			intToDouble(c_index, mat_ptr, size);
			floatToDouble(c_value, mat_ptr + size, size);
			mxSetCell(pa, i*sequenceAWidth + j, mat);

			c_index += size;
			c_value += size;
		}

	*_pa = pa;

	return true;
}