#include "common.h"

#include "utils.h"

bool generate_result(mxArray **_pa, const int sequenceAHeight, const int sequenceAWidth, const int *index_x, const int *index_y, const float *value, const int size)
{
	const int *c_index_x = index_x;
	const int *c_index_y = index_y;
	const float *c_value = value;

	mxArray *pa = mxCreateCellMatrix(sequenceAHeight, sequenceAWidth);
	if (!pa)
		return false;

	for (int i = 0; i<sequenceAHeight; i++)
		for (int j = 0; j < sequenceAWidth; j++)
		{
			mxArray *mat = mxCreateNumericMatrix(size, 3, mxDOUBLE_CLASS, mxREAL);
			if (!mat)
				return false;

			double *mat_ptr = mxGetPr(mat);
			convertArrayFromIntToDoublePlusOne(c_index_x, mat_ptr, size);
			convertArrayFromIntToDoublePlusOne(c_index_y, mat_ptr + size, size);
			convertArrayFromFloatToDouble(c_value, mat_ptr + 2 * size, size);
			mxSetCell(pa, i*sequenceAWidth + j, mat);

			c_index_x += size;
			c_index_y += size;
			c_value += size;
		}

	*_pa = pa;

	return true;
}

bool generatePaddedMatrix(mxArray **_pa, const int sequencePaddedHeight, const int sequencePaddedWidth, const float *data)
{
	mxArray *pa = mxCreateDoubleMatrix(sequencePaddedHeight, sequencePaddedWidth, mxREAL);
	if (!pa)
		return false;

	double *matPointer = mxGetPr(pa);
	convertArrayFromFloatToDouble(data, matPointer, sequencePaddedHeight * sequencePaddedWidth);

	*_pa = pa;

	return true;
}