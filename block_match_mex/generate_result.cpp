#include "common.h"

#include "utils.h"

template <typename IntermidateType, typename ResultType>
void generate_result(mxArray **_pa, const int sequenceAHeight, const int sequenceAWidth, const int *index_x, const int *index_y, 
	const IntermidateType *value, const int size)
{
	const int *c_index_x = index_x;
	const int *c_index_y = index_y;
	const IntermidateType *c_value = value;

	mxArray *pa = mxCreateCellMatrix(sequenceAHeight, sequenceAWidth);

	for (int i = 0; i<sequenceAHeight; i++)
		for (int j = 0; j < sequenceAWidth; j++)
		{
			mxArray *mat = mxCreateNumericMatrix(size, 3, getMxClassId(typeid(ResultType)), mxREAL);

			ResultType *mat_ptr = static_cast<ResultType*>(mxGetData(mat));
			convertArrayTypeAndPlusOne(c_index_x, mat_ptr, size);
			convertArrayTypeAndPlusOne(c_index_y, mat_ptr + size, size);
			convertArrayType(c_value, mat_ptr + 2 * size, size);
			mxSetCell(pa, i*sequenceAWidth + j, mat);

			c_index_x += size;
			c_index_y += size;
			c_value += size;
		}

	*_pa = pa;
}

template <typename IntermidateType, typename ResultType>
void generatePaddedMatrix(mxArray **_pa, const int sequencePaddedHeight, const int sequencePaddedWidth, const IntermidateType *data)
{
	mxArray *pa = mxCreateNumericMatrix(sequencePaddedHeight, sequencePaddedWidth, getMxClassId(typeid(ResultType)) ,mxREAL);

	ResultType *matPointer = static_cast<ResultType*>(mxGetData(pa));
	convertArrayType(data, matPointer, sequencePaddedHeight * sequencePaddedWidth);

	*_pa = pa;
}

template 
void generate_result<float,double>(mxArray **_pa, const int sequenceAHeight, const int sequenceAWidth, const int *index_x, const int *index_y,
	const float *value, const int size);
template
void generate_result<double, float>(mxArray **_pa, const int sequenceAHeight, const int sequenceAWidth, const int *index_x, const int *index_y,
	const double *value, const int size);
template
void generate_result<float, float>(mxArray **_pa, const int sequenceAHeight, const int sequenceAWidth, const int *index_x, const int *index_y,
	const float *value, const int size);
template
void generate_result<double, double>(mxArray **_pa, const int sequenceAHeight, const int sequenceAWidth, const int *index_x, const int *index_y,
	const double *value, const int size);
template
void generatePaddedMatrix<float,double>(mxArray **_pa, const int sequencePaddedHeight, const int sequencePaddedWidth, const float *data);
template
void generatePaddedMatrix<double, double>(mxArray **_pa, const int sequencePaddedHeight, const int sequencePaddedWidth, const double *data);
template
void generatePaddedMatrix<float, float>(mxArray **_pa, const int sequencePaddedHeight, const int sequencePaddedWidth, const float *data);
template
void generatePaddedMatrix<double, float>(mxArray **_pa, const int sequencePaddedHeight, const int sequencePaddedWidth, const double *data);