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
#include <cstdint>
#include <typeindex>
#include <cstring>

template<typename T1, typename T2>
void copyArray(const T1 *src, T2*dst, size_t n)
{
	for (size_t i = 0; i<n; ++i)
	{
		dst[i] = static_cast<T2>(src[i]);
	}
}

void convertArrayDataType(void *input, double *out, size_t n, std::type_index in_type)
{
	if (in_type == typeid(bool))
		copyArray<bool, double>(static_cast<bool*>(input), out, n);
	else if (in_type == typeid(uint8_t))
		copyArray<uint8_t, double>(static_cast<uint8_t*>(input), out, n);
	else if (in_type == typeid(uint16_t))
		copyArray<uint16_t, double>(static_cast<uint16_t*>(input), out, n);
	else if (in_type == typeid(uint32_t))
		copyArray<uint32_t, double>(static_cast<uint32_t*>(input), out, n);
	else if (in_type == typeid(uint64_t))
		copyArray<uint64_t, double>(static_cast<uint64_t*>(input), out, n);
	else if (in_type == typeid(int8_t))
		copyArray<int8_t, double>(static_cast<int8_t*>(input), out, n);
	else if (in_type == typeid(int16_t))
		copyArray<int16_t, double>(static_cast<int16_t*>(input), out, n);
	else if (in_type == typeid(int32_t))
		copyArray<int32_t, double>(static_cast<int32_t*>(input), out, n);
	else if (in_type == typeid(int64_t))
		copyArray<int64_t, double>(static_cast<int64_t*>(input), out, n);
	else if (in_type == typeid(float))
		copyArray<float, double>(static_cast<float*>(input), out, n);
	else if (in_type == typeid(double))
		memcpy(out, input, n * sizeof(double));
	else
		throw std::exception();
}

std::type_index getTypeIndexFromMxClassId(mxClassID classId)
{
	if (classId == mxLOGICAL_CLASS)
		return typeid(bool);
	else if (classId == mxUINT8_CLASS)
		return typeid(uint8_t);
	else if (classId == mxUINT16_CLASS)
		return typeid(uint16_t);
	else if (classId == mxUINT32_CLASS)
		return typeid(uint32_t);
	else if (classId == mxUINT64_CLASS)
		return typeid(uint64_t);
	else if (classId == mxSINGLE_CLASS)
		return typeid(float);
	else if (classId == mxDOUBLE_CLASS)
		return typeid(double);
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
	else if (type == typeid(int8_t))
		return sizeof(int8_t);
	else if (type == typeid(int16_t))
		return sizeof(int16_t);
	else if (type == typeid(int32_t))
		return sizeof(int32_t);
	else if (type == typeid(int64_t))
		return sizeof(int64_t);
	else if (type == typeid(float))
		return sizeof(float);
	else if (type == typeid(double))
		return sizeof(double);
	else
		throw std::exception();
}

void generateResult(const mxArray *result, const mxArray *index, mxArray **out_)
{
	size_t ndimsResult = mxGetNumberOfDimensions(result);
	if (ndimsResult != 3)
		throw std::exception();
	const size_t *resultdims = mxGetDimensions(result);
	if (index)
	{
		size_t ndimsIndex = mxGetNumberOfDimensions(index);
		if (ndimsIndex != 4)
			throw std::exception();
		const size_t *indexdims = mxGetDimensions(index);
		if (indexdims[0] != resultdims[0] || indexdims[1] != 2 || indexdims[2] != resultdims[1] || indexdims[3] != resultdims[2])
			throw std::exception();
	}

	std::type_index resultType = getTypeIndexFromMxClassId(mxGetClassID(result));
	std::type_index indexType = typeid(nullptr);
	if (index)
		indexType = getTypeIndexFromMxClassId(mxGetClassID(index));
	size_t resultElementSize = getDataTypeElementSize(resultType);
	size_t indexElementSize = 0;
	if (index)
		indexElementSize = getDataTypeElementSize(indexType);

	char *resultPtr = static_cast<char*>(mxGetData(result));
	char *indexXPtr = 0, *indexYPtr = 0;
	size_t indexPtrOffset = resultdims[0] * indexElementSize;
	if (index) {
		indexXPtr = static_cast<char*>(mxGetData(index));
		indexYPtr = indexXPtr + indexPtrOffset;
	}
	
	const size_t outCellDims[2] = { resultdims[1], resultdims[2] };
	mxArray *out = mxCreateCellArray(2, outCellDims);

	for (size_t i_dim1 = 0;i_dim1<resultdims[2];i_dim1++)
	{
		for (size_t i_dim0=0;i_dim0<resultdims[1];i_dim0++)
		{
			mxArray *currentMatrix;
			if (index)
				currentMatrix = mxCreateDoubleMatrix(resultdims[0], 3, mxREAL);
			else
				currentMatrix = mxCreateDoubleMatrix(resultdims[0], 1, mxREAL);

			double *currentMatrixPtr = mxGetPr(currentMatrix);
			convertArrayDataType(resultPtr, currentMatrixPtr, resultdims[0], resultType);
			resultPtr += resultElementSize * resultdims[0];
			currentMatrixPtr += resultdims[0];
			if (index)
			{
				convertArrayDataType(indexXPtr, currentMatrixPtr, resultdims[0], indexType);
				indexXPtr += 2 * indexPtrOffset;
				currentMatrixPtr += resultdims[0];
				convertArrayDataType(indexYPtr, currentMatrixPtr, resultdims[0], indexType);
				indexYPtr += 2 * indexPtrOffset;
			}
			
			size_t currentCellIndex[2] = { i_dim0, i_dim1 };
			mxSetCell(out, mxCalcSingleSubscript(out, 2, currentCellIndex),currentMatrix);
		}
	}
	*out_ = out;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	if (!(nrhs == 1 || nrhs == 2))
		mexErrMsgTxt("Invalid input");
	if (nlhs != 1)
		mexErrMsgTxt("Invalid output");
	try {
		if (nrhs == 1)
			generateResult(prhs[0], nullptr, &plhs[0]);
		else if (nrhs == 2)
			generateResult(prhs[0], prhs[1], &plhs[0]);
		else
			throw std::exception();
	}
	catch (std::exception &)
	{
		mexErrMsgTxt("Invalid input");
	}
}