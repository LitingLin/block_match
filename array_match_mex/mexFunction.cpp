#include "common.h"

/*
* plhs
* [0]: Result
* [1]: Index
*/
template <typename IntermidateType>
void process(ArrayMatchMexContext *context, int nlhs, mxArray *plhs[])
{
	try {
		std::type_index indexDataType = typeid(nullptr);
		if (nlhs > 1)
			indexDataType = context->indexDataType;

		ArrayMatch<IntermidateType> arrayMatch(context->sourceAType, context->sourceBType,
			context->resultType, indexDataType,
			context->method,
			context->sort,
			context->numberOfArrayA, context->numberOfArrayB,
			context->lengthOfArray,
			context->retain
		);
		int dim0 = context->numberOfArrayA, dim1 = context->numberOfArrayB;
		mxMatrixAllocator<size_t, size_t> matrixC(context->resultType, dim0, dim1);
		if (nlhs <= 1)
			dim0 = dim1 = 0;
		mxMatrixAllocator<size_t, size_t> index(context->indexDataType, dim1, dim0);
		
		size_t maxMxMemorySize = matrixC.getSize() + index.getSize();

		try {
			arrayMatch.initialize();
		}
		catch (memory_alloc_exception &exp) {
			std::string message = exp.what();
			message = message + "MATLAB:\t" + std::to_string(0) + "\t" + std::to_string(maxMxMemorySize) + "\n";
			mexErrMsgTxt(message.c_str());
			return;
		}
		try {
			matrixC.alloc();
			if (nlhs > 1)
				index.alloc();
		}
		catch (...)
		{
			size_t currentMxMemorySize = 0;
			if (matrixC.isAllocated())
				currentMxMemorySize += matrixC.getSize();
			if (index.isAllocated())
				currentMxMemorySize += index.getSize();
			reportMemoryAllocationFailed(currentMxMemorySize, maxMxMemorySize);
			throw;
		}

		arrayMatch.execute(context->A, context->B,
			matrixC.getData(),
			index.getData()
		);

		plhs[0] = matrixC.release();

		if (nlhs > 1)
			plhs[1] = index.release();
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
	struct ArrayMatchMexContext context;
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
