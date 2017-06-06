#include "lib_match_internal.h"

template <typename Type>
BlockMatch<Type>::~BlockMatch()
{
	if (m_instance)
		destroy();
}

template <typename Type>
void BlockMatch<Type>::destroy()
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type> *>(m_instance);
	delete instance;

	CUDA_CHECK_POINT(cudaSetDevice(0));

	m_instance = nullptr;
}

template
LIB_MATCH_EXPORT
BlockMatch<float>::~BlockMatch();
template
LIB_MATCH_EXPORT
BlockMatch<double>::~BlockMatch();

template
LIB_MATCH_EXPORT
void BlockMatch<float>::destroy();
template
LIB_MATCH_EXPORT
void BlockMatch<double>::destroy();