#include "lib_match_internal.h"

template <typename Type>
BlockMatchImpl<Type>::~BlockMatchImpl()
{
	if (m_instance)
		destroy();
}

template <typename Type>
void BlockMatchImpl<Type>::destroy()
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type> *>(m_instance);
	delete instance;

	m_instance = nullptr;
}

template
LIB_MATCH_EXPORT
BlockMatchImpl<float>::~BlockMatch();
template
LIB_MATCH_EXPORT
BlockMatchImpl<double>::~BlockMatch();

template
LIB_MATCH_EXPORT
void BlockMatchImpl<float>::destroy();
template
LIB_MATCH_EXPORT
void BlockMatchImpl<double>::destroy();