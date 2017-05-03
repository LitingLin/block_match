#include "common.h"
#include "utils.h"
#include <string.h>

mxClassID type_index_to_mx_class_id(std::type_index type)
{
	if (type == typeid(bool))
		return mxLOGICAL_CLASS;
	else if (type == typeid(uint8_t))
		return mxUINT8_CLASS;
	else if (type == typeid(int8_t))
		return mxINT8_CLASS;
	else if (type == typeid(uint16_t))
		return mxUINT16_CLASS;
	else if (type == typeid(int16_t))
		return mxINT16_CLASS;
	else if (type == typeid(uint32_t))
		return mxUINT32_CLASS;
	else if (type == typeid(int32_t))
		return mxINT32_CLASS;
	else if (type == typeid(uint64_t))
		return mxUINT64_CLASS;
	else if (type == typeid(float))
		return mxSINGLE_CLASS;
	else if (type == typeid(double))
		return mxDOUBLE_CLASS;

	throw std::runtime_error("Unknown type_index");
}

#include <array>

template <typename ...Types>
class mxMatrixAllocator
{
public:
	mxMatrixAllocator(std::type_index type, Types ...dimensions)
		: dims{ static_cast<size_t>(dimensions)... }, ptr(nullptr), classID(type_index_to_mx_class_id(type))
	{ }
	~mxMatrixAllocator()
	{
		if (ptr)
		{
			mxDestroyArray(ptr);
		}
	}
	void alloc()
	{
		ptr = mxCreateNumericArray(sizeof...(Types), dims.data(), classID, mxREAL);
	}
	void resetSize(Types ...dimensions)
	{
		dims = { static_cast<size_t>(dimensions)... };
	}
	mxArray *get() const
	{
		return ptr;
	}
	void *getData() const
	{
		if (!ptr)
			return nullptr;
		return mxGetData(ptr);
	}
	void reset()
	{
		mxDestroyArray(ptr);
		ptr = nullptr;
	}
	mxArray *release()
	{
		mxArray *ptr_ = this->ptr;
		this->ptr = nullptr;
		return ptr_;
	}
private:
	std::array<size_t, sizeof...(Types)> dims;
	mxArray *ptr;
	mxClassID classID;
};

/*
 * plhs
 * [0]: Result
 * [1]: Index
 * [2]: Padded_A
 * [3]: Padded_B
 */
template <typename IntermidateType>
void process(BlockMatchMexContext *context, int nlhs, mxArray *plhs[])
{
	try {
		std::type_index indexDataType = typeid(nullptr);
		if (nlhs > 1)
			indexDataType = context->indexDataType;

		BlockMatch<IntermidateType> blockMatch(context->sourceAType, context->sourceBType,
			context->resultType, indexDataType,
			context->searchType, context->method,
			context->padMethodA, context->padMethodB,
			context->sequenceABorderType, context->searchFrom,
			context->sort,
			context->sequenceAMatrixDimensions[0], context->sequenceAMatrixDimensions[1],
			context->sequenceBMatrixDimensions[0], context->sequenceBMatrixDimensions[1],
			context->searchRegion_M, context->searchRegion_N,
			context->block_M, context->block_N,
			context->sequenceAStride_M, context->sequenceAStride_N,
			context->sequenceBStride_M, context->sequenceBStride_N,
			context->sequenceAPadding_M_Pre, context->sequenceAPadding_M_Post,
			context->sequenceAPadding_N_Pre, context->sequenceAPadding_N_Post,
			context->sequenceBPadding_M_Pre, context->sequenceBPadding_M_Post,
			context->sequenceBPadding_N_Pre, context->sequenceBPadding_N_Post,
			context->retain
		);

		int dim0, dim1, dim2;
		blockMatch.get_matrixA_padded_dimensions(&dim0, &dim1);
		mxMatrixAllocator<size_t, size_t> matrixA_padded(context->sourceAType, dim1, dim0);
		blockMatch.get_matrixB_padded_dimensions(&dim0, &dim1);
		mxMatrixAllocator<size_t, size_t> matrixB_padded(context->sourceBType, dim1, dim0);
		blockMatch.get_matrixC_dimensions(&dim0, &dim1, &dim2);

		mxMatrixAllocator<size_t, size_t, size_t> matrixC(context->resultType, dim1, dim0, dim2);
		mxMatrixAllocator<size_t, size_t, size_t, size_t> index(context->indexDataType, dim1, dim0, dim2, 2);

		matrixC.alloc();
		ContiguousMemoryIterator matrixCIterator(matrixC.getData(), dim2);
		if (nlhs > 1)
			index.alloc();

		ContiguousMemoryIterator indexXIterator(index.getData(), dim2 * 2);
		ContiguousMemoryIterator indexYIterator(static_cast<char*>(index.getData()) + dim2*getTypeSize(context->indexDataType), dim2 * 2);
		if (nlhs > 2)
			matrixA_padded.alloc();
		if (nlhs > 3)
			matrixB_padded.alloc();

		void *matrixAPaddedPtr = nullptr;
		if (nlhs > 2)
			matrixAPaddedPtr = matrixA_padded.getData();
		void *matrixBPaddedPtr = nullptr;
		if (nlhs > 3)
			matrixBPaddedPtr = matrixA_padded.getData();

		ContiguousMemoryIterator *indexXIteratorPtr = nullptr;
		ContiguousMemoryIterator *indexYIteratorPtr = nullptr;
		if (nlhs > 1)
		{
			indexXIteratorPtr = &indexXIterator;
			indexYIteratorPtr = &indexYIterator;
		}
		blockMatch.executev2(context->sequenceAMatrixPointer, context->sequenceBMatrixPointer,
			&matrixCIterator,
			matrixAPaddedPtr,
			matrixBPaddedPtr,
			indexXIteratorPtr, indexYIteratorPtr
		);

		plhs[0] = matrixC.release();

		if (nlhs > 1)
			plhs[1] = index.release();

		if (nlhs > 2)
			plhs[2] = matrixA_padded.release();
		if (nlhs > 3)
			plhs[3] = matrixB_padded.release();
	}
	catch (std::runtime_error &exp)
	{
		mexErrMsgTxt(exp.what());
	}
	catch (std::bad_alloc &exp)
	{
		mexErrMsgTxt(exp.what());
	}
}

extern "C"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
	const mxArray *prhs[])
{
	libMatchMexInitalize();
	struct BlockMatchMexContext context;
	struct LibMatchMexErrorWithMessage errorMessage = parseParameter(&context, nlhs, plhs, nrhs, prhs);

	if (errorMessage.error != LibMatchMexError::success)
	{
		mexErrMsgTxt(errorMessage.message);
		return;
	}

	if (context.intermediateType == typeid(float))
		process<float>(&context, nlhs, plhs);
	else if (context.intermediateType == typeid(double))
		process<double>(&context, nlhs, plhs);
	else
		mexErrMsgTxt("Processing data type can only be float or double");
}
