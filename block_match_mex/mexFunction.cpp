#include "common.h"
#include <string.h>

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
			context->sequenceABorderType, 
			context->sort,
			context->sequenceAMatrixDimensions[0], context->sequenceAMatrixDimensions[1],
			context->sequenceBMatrixDimensions[0], context->sequenceBMatrixDimensions[1],
			context->sequenceAMatrixDimensions[2],
			context->searchRegion_M_pre, context->searchRegion_M_post,
			context->searchRegion_N_pre, context->searchRegion_N_post,
			context->block_M, context->block_N,
			context->sequenceAStride_M, context->sequenceAStride_N,
			context->sequenceBStride_M, context->sequenceBStride_N,
			context->sequenceAPadding_M_Pre, context->sequenceAPadding_M_Post,
			context->sequenceAPadding_N_Pre, context->sequenceAPadding_N_Post,
			context->sequenceBPadding_M_Pre, context->sequenceBPadding_M_Post,
			context->sequenceBPadding_N_Pre, context->sequenceBPadding_N_Post,
			context->retain,
			context->threshold,
			static_cast<IntermidateType>(context->thresholdValue),
			static_cast<IntermidateType>(context->thresholdReplacementValue),
			true,
			context->numberOfThreads,
			context->indexOfDevice
		);
		int dim0, dim1, dim2;
		blockMatch.get_matrixC_dimensions(&dim0, &dim1, &dim2);
		mxMatrixAllocator<size_t, size_t> matrixC(context->resultType, dim2, dim1 * dim0);
		if (nlhs <= 1)
			dim0 = dim1 = dim2 = 0;
		mxMatrixAllocator<size_t, size_t, size_t> index(context->indexDataType, dim2, 2, dim1 * dim0);
		blockMatch.get_matrixA_padded_dimensions(&dim0, &dim1, &dim2);
		if (nlhs <= 2)
			dim0 = dim1 = dim2 = 0;
		size_t input2Ddim0 = 0, input2Ddim1 = 0,
			input3ddim0 = 0, input3ddim1 = 0, input3ddim2 = 0;
		if (context->sequenceMatrixNumberOfDimensions == 2)
			input2Ddim0 = dim0, input2Ddim1 = dim1;
		else
			input3ddim0 = dim0, input3ddim1 = dim1, input3ddim2 = dim2;

		mxMatrixAllocator<size_t, size_t> matrixA_padded(context->sourceAType, input2Ddim1, input2Ddim0);
		mxMatrixAllocator<size_t, size_t, size_t> matrixA3D_padded(context->sourceAType, input3ddim1, input3ddim0, input3ddim2);
		
		blockMatch.get_matrixB_padded_dimensions(&dim0, &dim1, &dim2);
		if (nlhs <= 3)
			dim0 = dim1 = dim2 = 0;
		if (context->sequenceMatrixNumberOfDimensions == 2)
			input2Ddim0 = dim0, input2Ddim1 = dim1;
		else
			input3ddim0 = dim0, input3ddim1 = dim1, input3ddim2 = dim2;
		mxMatrixAllocator<size_t, size_t> matrixB_padded(context->sourceBType, input2Ddim1, input2Ddim0);
		mxMatrixAllocator<size_t, size_t, size_t> matrixB3D_padded(context->sourceBType, input3ddim1, input3ddim0, input3ddim2);

		size_t maxMxMemorySize = matrixA_padded.getSize() + matrixB_padded.getSize() + matrixC.getSize() + index.getSize() + 
			matrixA3D_padded.getSize() + matrixB3D_padded.getSize();

		try {
			blockMatch.initialize();
		}
		catch(memory_alloc_exception &exp){
			std::string message = exp.what();
			message = message + "MATLAB:\t" + std::to_string(0) + "\t" + std::to_string(maxMxMemorySize) + "\n";
			mexErrMsgTxt(message.c_str());
			return;
		}
		try {
			matrixC.alloc();
			if (nlhs > 1)
				index.alloc();
			if (nlhs > 2) {
				matrixA_padded.alloc();
				matrixA3D_padded.alloc();
			}
			if (nlhs > 3) {
				matrixB_padded.alloc();
				matrixB3D_padded.alloc();
			}
		}
		catch (...)
		{
			size_t currentMxMemorySize = 0;
			if (matrixC.isAllocated())
				currentMxMemorySize += matrixC.getSize();
			if (index.isAllocated())
				currentMxMemorySize += index.getSize();
			if (matrixA_padded.isAllocated()) {
				currentMxMemorySize += matrixA_padded.getSize();
				currentMxMemorySize += matrixA3D_padded.getSize();
			}
			if (matrixB_padded.isAllocated()) {
				currentMxMemorySize += matrixB_padded.getSize();
				currentMxMemorySize += matrixB3D_padded.getSize();
			}
			reportMemoryAllocationFailed(currentMxMemorySize, maxMxMemorySize);
			throw;
		}
		blockMatch.get_matrixC_dimensions(&dim0, &dim1, &dim2);
		ContiguousMemoryIterator matrixCIterator(matrixC.getData(), dim2 * getTypeSize(context->resultType));
		ContiguousMemoryIterator indexXIterator(static_cast<char*>(index.getData()) + dim2*getTypeSize(context->indexDataType),
			dim2 * 2 * getTypeSize(context->indexDataType));
		ContiguousMemoryIterator indexYIterator(index.getData(), dim2 * 2 * getTypeSize(context->indexDataType));

		void *matrixAPaddedPtr = nullptr;
		if (nlhs > 2) {
			if (context->sequenceMatrixNumberOfDimensions == 2)
				matrixAPaddedPtr = matrixA_padded.getData();
			else
				matrixAPaddedPtr = matrixA3D_padded.getData();
		}
		void *matrixBPaddedPtr = nullptr;
		if (nlhs > 3) {
			if (context->sequenceMatrixNumberOfDimensions == 2)
				matrixBPaddedPtr = matrixB_padded.getData();
			else
				matrixBPaddedPtr = matrixB3D_padded.getData();
		}

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

		if (nlhs > 2) {
			if (context->sequenceMatrixNumberOfDimensions == 2)
				plhs[2] = matrixA_padded.release();
			else
				plhs[2] = matrixA3D_padded.release();
		}
		if (nlhs > 3) {
			if (context->sequenceMatrixNumberOfDimensions == 2)
				plhs[3] = matrixB_padded.release();
			else
				plhs[3] = matrixB3D_padded.release();
		}
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
		mexErrMsgTxt("Computing data type can only be float or double");
}
