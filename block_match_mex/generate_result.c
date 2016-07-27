#include "common.h"

#include "utils.h"

bool generate_result(struct LibBlockMatchMexContext *context,mxArray **_pa, const int *index, const float *value)
{
	const int *c_index = index;
	const float *c_value = value;

	const int blockHeight = context->blockHeight;
	const int blockWidth = context->blockWidth;
	const int sequenceBPaddingHeight = context->sequenceBPaddingHeight;
	const int sequenceBPaddingWidth = context->sequenceBPaddingWidth;
	const int sequenceBStrideHeight = context->sequenceBStrideHeight;
	const int sequenceBStrideWidth = context->sequenceBStrideWidth;

	const int sequenceAHeight = (context->sequenceAMatrixDimensions[0] + 2 * context->sequenceAPaddingHeight - context->blockHeight) / context->sequenceAStrideHeight + 1,
		sequenceAWidth = (context->sequenceAMatrixDimensions[1] + 2 * context->sequenceAPaddingWidth - context->blockWidth) / context->sequenceAStrideWidth + 1;
	const int sequenceBHeight = (context->sequenceBMatrixDimensions[0] + 2 * sequenceBPaddingHeight - blockHeight) / sequenceBStrideHeight + 1;
	const int sequenceBWidth = (context->sequenceBMatrixDimensions[1] + 2 * sequenceBPaddingWidth - blockWidth) / sequenceBStrideWidth + 1;
	int size = context->retain;
	if (!size)
		size = sequenceBHeight * sequenceBWidth;
	
	mxArray *pa = mxCreateCellMatrix(sequenceAHeight, sequenceAWidth);
	for (int i=0;i<sequenceAHeight;i++)
		for (int j = 0; j < sequenceAWidth; j++)
		{
			mxArray *mat = mxCreateNumericMatrix(size, 3, mxDOUBLE_CLASS, mxREAL);
			if (!mat)
				return false;

			double *mat_ptr = mxGetPr(mat);
			double *mat_ptr_x = mat_ptr;
			double *mat_ptr_y = mat_ptr + size;
			for (int k=0;k<size;++k)
			{
				int index_value = *c_index;
				*mat_ptr_x++ = (index_value % sequenceBHeight) *sequenceBStrideHeight - sequenceBPaddingHeight + 1;
				*mat_ptr_y++ = (index_value / sequenceBHeight) *sequenceBStrideWidth - sequenceBPaddingWidth + 1;
				++c_index;
			}
			//intToDouble(c_index, mat_ptr, size);
			floatToDouble(c_value, mat_ptr + 2* size, size);
			mxSetCell(pa, i*sequenceAWidth + j, mat);

			//c_index += size;
			c_value += size;
		}

	*_pa = pa;

	return true;
}