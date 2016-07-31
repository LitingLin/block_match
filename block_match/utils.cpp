int getLength(int matSize, int paddingSize, int blockSize, int strideSize)
{
	return (matSize + 2 * paddingSize - blockSize + 1) / strideSize;
}

int determineEndOfIndex(int matSize, int paddingSize, int blockSize)
{
	return matSize + paddingSize - blockSize + 1;
}

void generateIndexSequence(int *index, int size)
{
	for (int i=0;i<size;++i)
	{
		index[i] = i;
	}
}

template <typename TypeA, typename TypeB>
void type_convert(TypeA *a, TypeB *b, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		a[i] = b[i];
}

