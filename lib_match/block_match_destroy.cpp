#include "lib_match_internal.h"

template <typename Type>
void BlockMatch<Type>::destroy()
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type> *>(m_instance);
	delete instance;

	m_instance = nullptr;
}

LIB_MATCH_EXPORT
template
void BlockMatch<float>::destroy();
LIB_MATCH_EXPORT
template
void BlockMatch<double>::destroy();