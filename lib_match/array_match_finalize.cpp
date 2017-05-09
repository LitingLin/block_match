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
