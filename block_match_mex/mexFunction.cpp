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
}

template <size_t size, typename ...dimensions>
class mx_matrix_allocator
{
public:
	mx_matrix_allocator(std::type_index type, dimensions...)
		: ptr(nullptr), classID(type_index_to_mx_class_id(type)), dims({dimensions::info})
	{ }
	~mx_matrix_allocator()
	{
		if (ptr)
		{
			mxDestroyArray(ptr);
		}
	}
	void alloc()
	{
		if (m && n) {
			size_t dims[2] = { static_cast<size_t>(m),static_cast<size_t>(n) };
			ptr = mxCreateNumericArray(2, dims, classID, mxREAL);
		}
	}
	void resetSize(dimensions...)
	{
		this->m = m;
		this->n = n;
	}
	mxArray *get() const
	{
		return ptr;
	}
	void *getData() const
	{
		return mxGetData(ptr);
	}
	mxArray *release()
	{
		mxArray *ptr = this->ptr;
		this->ptr = nullptr;
		return ptr;
	}
private:
	std::array<size_t, size> dims;
	mxArray *ptr;
	mxClassID classID;
};

class mx_2d_matrix_allocator
{
public:
	mx_2d_matrix_allocator(std::type_index type, int m = 0, int n = 0)
		: m(m), n(n), ptr(nullptr), classID(type_index_to_mx_class_id(type))
	{ }
	~mx_2d_matrix_allocator()
	{
		if (ptr)
		{
			mxDestroyArray(ptr);
		}
	}
	void alloc()
	{
		if (m && n) {
			size_t dims[2] = { static_cast<size_t>(m),static_cast<size_t>(n) };
			ptr = mxCreateNumericArray(2, dims, classID, mxREAL);
		}
	}
	void resetSize(int m, int n)
	{
		this->m = m;
		this->n = n;
	}
	mxArray *get() const
	{
		return ptr;
	}
	void *getData() const
	{
		return mxGetData(ptr);
	}
	mxArray *release()
	{
		mxArray *ptr = this->ptr;
		this->ptr = nullptr;
		return ptr;
	}
private:
	int m;
	int n;
	mxArray *ptr;
	mxClassID classID;
};

class mx_3d_matrix_allocator
{
public:
	mx_3d_matrix_allocator(std::type_index type, int dim0 = 0, int dim1 = 0)
		: dim0(dim0), dim1(dim1), ptr(nullptr), classID(type_index_to_mx_class_id(type))
	{ }
	~mx_3d_matrix_allocator()
	{
		if (ptr)
		{
			mxDestroyArray(ptr);
		}
	}
	void alloc()
	{
		if (dim0 && dim1) {
			size_t dims[2] = { static_cast<size_t>(dim1),static_cast<size_t>(dim0) };
			ptr = mxCreateNumericArray(2, dims, classID, mxREAL);
		}
	}
	void resetSize(int m, int n)
	{
		this->dim0 = m;
		this->dim1 = n;
	}
	mxArray *get() const
	{
		return ptr;
	}
	void *getData() const
	{
		return mxGetData(ptr);
	}
	mxArray *release()
	{
		mxArray *ptr = this->ptr;
		this->ptr = nullptr;
		return ptr;
	}
private:
	int dim0;
	int dim1;
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
	int sequenceASize = context->sequenceAMatrixDimensions[0] * context->sequenceAMatrixDimensions[1];
	int sequenceBSize = context->sequenceBMatrixDimensions[0] * context->sequenceBMatrixDimensions[1];
	BlockMatch<IntermidateType> blockMatch(context->sourceAType, context->sourceBType,
		context->resultType, context->indexDataType,
		context->searchType, context->method,
		context->padMethodA, context->padMethodB,
		context->sequenceABorderType, context->searchFrom,
		context->sort,
		context->sequenceAMatrixDimensions[0], context->sequenceAMatrixDimensions[1],
		context->sequenceBMatrixDimensions[0], context->sequenceBMatrixDimensions[1],
		context->searchRegion_M,context->searchRegion_N,
		context->block_M, context->block_N,
		context->sequenceAStride_M,context->sequenceAStride_N,
		context->sequenceBStride_M, context->sequenceBStride_N,
		context->sequenceAPadding_M_Pre, context->sequenceAPadding_M_Post,
		context->sequenceAPadding_N_Pre, context->sequenceAPadding_N_Post,
		context->sequenceBPadding_M_Pre, context->sequenceBPadding_M_Post,
		context->sequenceBPadding_N_Pre, context->sequenceBPadding_N_Post,
		context->retain
		);
	int dim0, dim1, dim2;
	blockMatch.get_matrixA_padded_dimensions(&dim0, &dim1);
	mx_2d_matrix_allocator matrixA_padded(context->sourceAType, dim1, dim0);
	blockMatch.get_matrixB_padded_dimensions(&dim0, &dim1);
	mx_2d_matrix_allocator matrixB_padded(context->sourceBType, dim1, dim0);
	blockMatch.get_matrixC_dimensions(&dim0, &dim1, &dim2);
	
	system_memory_allocator<IntermidateType> sequenceAPointer_converted(sequenceASize);
	system_memory_allocator<IntermidateType> sequenceBPointer_converted(sequenceASize);

	int matrixC_M = libMatchWarper.matrixC_M;
	int matrixC_N = libMatchWarper.matrixC_N;
	int matrixC_O = libMatchWarper.matrixC_O;
	int matrixA_padded_M = libMatchWarper.matrixA_padded_M;
	int matrixA_padded_N = libMatchWarper.matrixA_padded_N;
	int matrixB_padded_M = libMatchWarper.matrixB_padded_M;
	int matrixB_padded_N = libMatchWarper.matrixB_padded_N;

	system_memory_allocator<IntermidateType> matrixC(matrixC_M * matrixC_N * matrixC_O);
	system_memory_allocator<IntermidateType> matrixA_padded(matrixA_padded_M * matrixA_padded_N);
	system_memory_allocator<IntermidateType> matrixB_padded(matrixB_padded_M * matrixB_padded_N);
	system_memory_allocator<int> index_x(matrixC_M * matrixC_N * matrixC_O);
	system_memory_allocator<int> index_y(matrixC_M * matrixC_N * matrixC_O);

	try {
		sequenceAPointer_converted.alloc(); sequenceBPointer_converted.alloc();
		matrixC.alloc();
		matrixA_padded.alloc(); matrixB_padded.alloc();
		index_x.alloc(); index_y.alloc();

		convertArrayType(context->sourceAType, context->intermediateType, context->sequenceAMatrixPointer, sequenceAPointer_converted.get(), sequenceASize);
		convertArrayType(context->sourceBType, context->intermediateType, context->sequenceBMatrixPointer, sequenceBPointer_converted.get(), sequenceBSize);

		libMatchWarper.execute(sequenceAPointer_converted, sequenceBPointer_converted, matrixC, matrixA_padded, matrixB_padded, index_x, index_y);

		libMatchWarper.destroy();
	}
	catch (memory_alloc_exception &exp)
	{
		
	}
	catch (page_locked_memory_alloc_exception &exp)
	{
		
	}
	catch (gpu_memory_alloc_exception &exp)
	{
		
	}
	catch (std::exception &exp)
	{
		mexErrMsgTxt(exp.what());
	}
	sequenceAPointer_converted.release();
	sequenceBPointer_converted.release();
	generate_result<IntermidateType, ResultType>(&plhs[0], matrixC_N, matrixC_M, index_y.get(), index_x.get(), matrixC.get(), matrixC_O);

	index_y.release();
	index_x.release();
	matrixC.release();

	if (nlhs > 1)
	{
		generatePaddedMatrix<IntermidateType, ResultType>(&plhs[1], matrixA_padded_N, matrixA_padded_M, matrixA_padded.get());
	}

	matrixA_padded.release();

	if (nlhs > 2)
	{
		generatePaddedMatrix<IntermidateType, ResultType>(&plhs[2], matrixB_padded_N, matrixB_padded_M, matrixB_padded.get());
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
