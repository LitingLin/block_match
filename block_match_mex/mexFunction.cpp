#include "common.h"
#include "utils.h"
#include <string.h>

template <typename IntermidateType>
class LibBlockMatchWarper
{
public:
	LibBlockMatchWarper()
		: m_isDestroyed(true)
	{
	}
	void init(BlockMatchMexContext *context)
	{
		blockMatchInitialize<IntermidateType>(&instance,
			context->searchType,
			context->method,
			context->padMethodA, context->padMethodB,
			context->sequenceABorderType,
			context->searchFrom,
			context->sort,
			context->sequenceAMatrixDimensions[1], context->sequenceAMatrixDimensions[0], context->sequenceBMatrixDimensions[1], context->sequenceBMatrixDimensions[0],
			context->searchRegionWidth, context->searchRegionHeight,
			context->blockWidth, context->blockHeight,
			context->sequenceAStrideWidth, context->sequenceAStrideHeight,
			context->sequenceBStrideWidth, context->sequenceBStrideHeight,
			context->sequenceAPaddingWidthPre, context->sequenceAPaddingWidthPost,
			context->sequenceAPaddingHeightPre, context->sequenceAPaddingHeightPost,
			context->sequenceBPaddingWidthPre, context->sequenceBPaddingWidthPost,
			context->sequenceBPaddingHeightPre, context->sequenceBPaddingHeightPost,
			context->retain,
			&matrixC_M, &matrixC_N, &matrixC_O,
			&matrixA_padded_M, &matrixA_padded_N,
			&matrixB_padded_M, &matrixB_padded_N);
		m_isDestroyed = false;
	}
	void execute(system_memory_allocator<IntermidateType> &sequenceAPointer_converted, system_memory_allocator<IntermidateType> &sequenceBPointer_converted,
		system_memory_allocator<IntermidateType> &matrixC,
		system_memory_allocator<IntermidateType> &matrixA_padded, system_memory_allocator<IntermidateType> &matrixB_padded,
		system_memory_allocator<int> &index_x, system_memory_allocator<int> &index_y)
	{
		blockMatchExecute(instance, sequenceAPointer_converted.get(), sequenceBPointer_converted.get(),
			matrixC.get(), matrixA_padded.get(), matrixB_padded.get(), index_x.get(), index_y.get());
	}
	~LibBlockMatchWarper()
	{
		if (!m_isDestroyed)
			blockMatchFinalize<IntermidateType>(instance);
	}
	void destroy()
	{
		blockMatchFinalize<IntermidateType>(instance);
		m_isDestroyed = true;
	}
	bool isDestroyed()
	{
		return m_isDestroyed;
	}

	bool m_isDestroyed;
	void *instance;
	int matrixC_M;
	int matrixC_N;
	int matrixC_O;
	int matrixA_padded_M;
	int matrixA_padded_N;
	int matrixB_padded_M;
	int matrixB_padded_N;
};

template <typename IntermidateType, typename ResultType>
void process(BlockMatchMexContext *context, int nlhs, mxArray *plhs[])
{
	int sequenceASize = context->sequenceAMatrixDimensions[0] * context->sequenceAMatrixDimensions[1];
	int sequenceBSize = context->sequenceBMatrixDimensions[0] * context->sequenceBMatrixDimensions[1];
	LibBlockMatchWarper<IntermidateType> libMatchWarper;
	try {
		libMatchWarper.init(context);

	}
	catch (std::exception &exp)
	{
		mexErrMsgTxt(exp.what());
	}
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

	/*errorMessage = validateParameter(&context);

	if (errorMessage.error != LibMatchMexError::success)
	{
		mexErrMsgTxt(errorMessage.message);
		return;
	}*/

	if (context.intermediateType == typeid(float) && context.resultType == typeid(double))
		process<float, double>(&context, nlhs, plhs);
	else if (context.intermediateType == typeid(float) && context.resultType == typeid(float))
		process<float, float>(&context, nlhs, plhs);
	else if (context.intermediateType == typeid(double) && context.resultType == typeid(float))
		process<double, float>(&context, nlhs, plhs);
	else if (context.intermediateType == typeid(double) && context.resultType == typeid(double))
		process<double, double>(&context, nlhs, plhs);
	else
		mexErrMsgTxt("Processing data type can only be float or double");
}
