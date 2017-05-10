#include "lib_match_internal.h"

#include "lib_match.h"

template <typename Type>
ArrayMatch<Type>::~ArrayMatch()
{
	destroy();
}

template <typename Type>
void ArrayMatch<Type>::destroy()
{
	if (m_instance) {
		ArrayMatchContext<Type> *instance = static_cast<ArrayMatchContext<Type> *>(m_instance);
		delete instance;
		m_instance = nullptr;
	}
}

template
LIB_MATCH_EXPORT
ArrayMatch<float>::~ArrayMatch();
template
LIB_MATCH_EXPORT
ArrayMatch<double>::~ArrayMatch();

template
LIB_MATCH_EXPORT
void ArrayMatch<float>::destroy();
template
LIB_MATCH_EXPORT
void ArrayMatch<double>::destroy();