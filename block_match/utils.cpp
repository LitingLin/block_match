

template <typename TypeA, typename TypeB>
void type_convert(TypeA *a, TypeB *b, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		a[i] = b[i];
}

