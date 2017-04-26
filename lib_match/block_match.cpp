#include "lib_match_internal.h"

template <typename Type>
void BlockMatch<Type>::get_matrixC_dimensions(int* dim0, int* dim1, int* dim2)
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type>*>(m_instance);
	*dim0 = instance->C_dimensions[0];
	*dim1 = instance->C_dimensions[1];
	*dim2 = instance->C_dimensions[2];
}

template <typename Type>
void BlockMatch<Type>::get_matrixA_padded_dimensions(int* m, int* n)
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type>*>(m_instance);

	*m = instance->matrixA_padded_M;
	*n = instance->matrixA_padded_N;
}

template <typename Type>
void BlockMatch<Type>::get_matrixB_padded_dimensions(int* m, int* n)
{
	BlockMatchContext<Type> *instance = static_cast<BlockMatchContext<Type>*>(m_instance);

	*m = instance->matrixB_padded_M;
	*n = instance->matrixB_padded_N;
}

template
LIB_MATCH_EXPORT
void BlockMatch<float>::get_matrixC_dimensions(int *, int *, int *);
template
LIB_MATCH_EXPORT
void BlockMatch<double>::get_matrixC_dimensions(int *, int *, int *);
template
LIB_MATCH_EXPORT
void BlockMatch<float>::get_matrixA_padded_dimensions(int *, int *);
template
LIB_MATCH_EXPORT
void BlockMatch<double>::get_matrixA_padded_dimensions(int *, int *);
template
LIB_MATCH_EXPORT
void BlockMatch<float>::get_matrixB_padded_dimensions(int *, int *);
template
LIB_MATCH_EXPORT
void BlockMatch<double>::get_matrixB_padded_dimensions(int *, int *);