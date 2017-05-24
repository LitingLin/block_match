#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define DLL_EXPORT_SYM __attribute__ ((dllexport))
#else
#define DLL_EXPORT_SYM __declspec(dllexport)
#endif
#else
#define DLL_EXPORT_SYM __attribute__ ((visibility ("default")))
#endif

#include <mex.h>
#include <matrix.h>
#include <exception>
#include <string.h>
#include <typeindex>
#include <stdint.h>
#include <limits>
#include <algorithm>

void getMaxIndexValue(const mxArray *pa, size_t *x, size_t *y)
{
	size_t max_x = 0, max_y = 0;
	size_t nDims = mxGetNumberOfDimensions(pa);
	if (nDims != 2)
		throw std::exception();
	const size_t *dimensions = mxGetDimensions(pa);

	for (size_t i_dim1 = 0; i_dim1 < dimensions[1]; i_dim1++) // column
	{
		for (size_t i_dim0 = 0; i_dim0 < dimensions[0]; i_dim0++) // row
		{
			size_t currentCellIndex[2] = { i_dim0, i_dim1 };
			mxArray *cell = mxGetCell(pa, mxCalcSingleSubscript(pa, 2, currentCellIndex));
			size_t nMatrixDims = mxGetNumberOfDimensions(cell);
			if (nMatrixDims != 2)
				throw std::exception();
			const size_t *matrixDimensions = mxGetDimensions(cell);
			if (matrixDimensions[1] != 3)
				throw std::exception();
			if (mxGetClassID(cell) != mxDOUBLE_CLASS)
				throw std::exception();
			double *matrixPtr = mxGetPr(cell);

			double *index_x_ptr = matrixPtr + matrixDimensions[0];
			double *index_y_ptr = index_x_ptr + matrixDimensions[0];
			double *end_ptr = index_y_ptr + matrixDimensions[0];

			for (double *c_ptr = index_x_ptr;c_ptr!= index_y_ptr;++c_ptr)
			{
				if (*c_ptr > max_x)
					max_x = static_cast<size_t>(*c_ptr);
			}
			for (double *c_ptr = index_y_ptr; c_ptr != end_ptr; ++c_ptr)
			{
				if (*c_ptr > max_y)
					max_y = static_cast<size_t>(*c_ptr);
			}
		}
	}
}

void checkSubMatrix(const mxArray *matrix, const size_t ndims, const size_t *dims, const mxClassID type)
{
	size_t nMatrixDims = mxGetNumberOfDimensions(matrix);
	if (nMatrixDims != ndims)
		throw std::exception();
	const size_t *matrixDimensions = mxGetDimensions(matrix);
	for (size_t i = 0; i < ndims; ++i)
		if (matrixDimensions[i] != dims[i])
			throw std::exception();
	if (mxGetClassID(matrix) != type)
		throw std::exception();
}

void generateResult(const mxArray *pa, mxArray **out_)
{
	size_t nDims = mxGetNumberOfDimensions(pa);
	if (nDims != 2)
		throw std::exception();
	const size_t *dimensions = mxGetDimensions(pa);

	mxArray *cell = mxGetCell(pa, 0);
	size_t nMatrixDims = mxGetNumberOfDimensions(cell);
	if (nMatrixDims != 2)
		throw std::exception();
	const size_t *matrixDimensions = mxGetDimensions(cell);
	if (matrixDimensions[1] != 3)
		throw std::exception();
	if (mxGetClassID(cell) != mxDOUBLE_CLASS)
		throw std::exception();

	size_t outmatrixDims[3] = { dimensions[0],dimensions[1], matrixDimensions[0] };
	mxArray *out = mxCreateNumericArray(3, outmatrixDims, mxDOUBLE_CLASS, mxREAL);
	double *out_ptr = mxGetPr(out);

	for (size_t i_dim1 = 0; i_dim1 < dimensions[1]; i_dim1++) // column
	{
		for (size_t i_dim0 = 0; i_dim0 < dimensions[0]; i_dim0++) // row
		{
			size_t currentCellIndex[2] = { i_dim0, i_dim1 };
			mxArray *currentMatrix = mxGetCell(pa, mxCalcSingleSubscript(pa, 2, currentCellIndex));
			checkSubMatrix(currentMatrix, 2, matrixDimensions, mxDOUBLE_CLASS);
			double *currentMatrixPtr = mxGetPr(currentMatrix);

			memcpy(out_ptr, currentMatrixPtr, matrixDimensions[0] * sizeof(double));
			out_ptr += matrixDimensions[0];
		}
	}
	*out_ = out;
}

template<typename T1, typename T2>
void copyArray(const T1 *src, T2*dst, size_t n)
{
	for (size_t i=0;i<n;++i)
	{
		dst[i] = static_cast<T2>(src[i]);
	}
}

void convertArrayDataType(double *input, void *out, size_t n, std::type_index out_type)
{
	if (out_type == typeid(bool))
		copyArray<double, bool>(input, static_cast<bool*>(out), n);
	else if (out_type == typeid(uint8_t))
		copyArray<double, uint8_t>(input, static_cast<uint8_t*>(out), n);
	else if (out_type == typeid(uint16_t))
		copyArray<double, uint16_t>(input, static_cast<uint16_t*>(out), n);
	else if (out_type == typeid(uint32_t))
		copyArray<double, uint32_t>(input, static_cast<uint32_t*>(out), n);
	else if (out_type == typeid(uint64_t))
		copyArray<double, uint64_t>(input, static_cast<uint64_t*>(out), n);
	else
		throw std::exception();
}

mxClassID getMxClassIdFromTypeIndex(std::type_index type_index)
{
	if (type_index == typeid(bool))
		return mxLOGICAL_CLASS;
	else if (type_index == typeid(uint8_t))
		return mxUINT8_CLASS;
	else if (type_index == typeid(uint16_t))
		return mxUINT16_CLASS;
	else if (type_index == typeid(uint32_t))
		return mxUINT32_CLASS;
	else if (type_index == typeid(uint64_t))
		return mxUINT64_CLASS;
	else
		throw std::exception();
}

size_t getDataTypeElementSize(std::type_index type)
{
	if (type == typeid(bool))
		return sizeof(bool);
	else if (type == typeid(uint8_t))
		return sizeof(uint8_t);
	else if (type == typeid(uint16_t))
		return sizeof(uint16_t);
	else if (type == typeid(uint32_t))
		return sizeof(uint32_t);
	else if (type == typeid(uint64_t))
		return sizeof(uint64_t);
	else
		throw std::exception();
}

void generateIndex(const mxArray *pa, mxArray **out_, size_t max)
{
	std::type_index out_type = typeid(nullptr);
#ifdef _MSC_VER
#pragma warning(disable:4804)
#endif
	if (max <= std::numeric_limits<bool>::max())
#ifdef _MSC_VER
#pragma warning(default:4804)
#endif
		out_type = typeid(bool);
	else if (max <= std::numeric_limits<uint8_t>::max())
		out_type = typeid(uint8_t);
	else if (max <= std::numeric_limits<uint16_t>::max())
		out_type = typeid(uint16_t);
	else if (max <= std::numeric_limits<uint32_t>::max())
		out_type = typeid(uint32_t);
	else if (max <= std::numeric_limits<uint64_t>::max())
		out_type = typeid(uint64_t);
	else
		throw std::exception();

	size_t nDims = mxGetNumberOfDimensions(pa);
	if (nDims != 2)
		throw std::exception();
	const size_t *dimensions = mxGetDimensions(pa);

	mxArray *cell = mxGetCell(pa, 0);
	size_t nMatrixDims = mxGetNumberOfDimensions(cell);
	if (nMatrixDims != 2)
		throw std::exception();
	const size_t *matrixDimensions = mxGetDimensions(cell);
	if (matrixDimensions[1] != 3)
		throw std::exception();
	if (mxGetClassID(cell) != mxDOUBLE_CLASS)
		throw std::exception();

	size_t outmatrixDims[4] = { dimensions[0],dimensions[1], matrixDimensions[0], 2 };
	mxArray *out = mxCreateNumericArray(4, outmatrixDims, getMxClassIdFromTypeIndex(out_type), mxREAL);
	char *out_ptr = (char*)mxGetData(out);
	size_t out_type_element_size = getDataTypeElementSize(out_type);

	for (size_t i_dim1 = 0; i_dim1 < dimensions[1]; i_dim1++) // column
	{
		for (size_t i_dim0 = 0; i_dim0 < dimensions[0]; i_dim0++) // row
		{
			size_t currentCellIndex[2] = { i_dim0, i_dim1 };
			mxArray *currentMatrix = mxGetCell(pa, mxCalcSingleSubscript(pa, 2, currentCellIndex));
			double *currentMatrixPtr = mxGetPr(currentMatrix);
			double *index_x_ptr = currentMatrixPtr + matrixDimensions[0];
			double *index_y_ptr = index_x_ptr + matrixDimensions[0];

			convertArrayDataType(index_x_ptr, out_ptr, matrixDimensions[0], out_type);
			out_ptr += matrixDimensions[0] * out_type_element_size;
			convertArrayDataType(index_y_ptr, out_ptr, matrixDimensions[0], out_type);
			out_ptr += matrixDimensions[0] * out_type_element_size;
		}
	}
	*out_ = out;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	if (nrhs != 1)
		mexErrMsgTxt("Invalid input");
	if (nlhs != 1 || nlhs != 2)
		mexErrMsgTxt("Invalid output");
	try {
		generateResult(prhs[0], &plhs[0]);
		if (nlhs == 2)
		{
			size_t max_x, max_y;
			getMaxIndexValue(prhs[0], &max_x, &max_y);
			generateIndex(prhs[0], &plhs[1], std::max(max_x, max_y));
		}
	}
	catch(std::exception &)
	{
		mexErrMsgTxt("Invalid input");
	}
}