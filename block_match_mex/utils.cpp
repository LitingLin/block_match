void convertArrayFromIntToDouble(const int *in, double *out, size_t n)
{
	for (size_t i=0;i<n;i++)
	{
		out[i] = in[i];
	}
}

void convertArrayFromIntToDoublePlusOne(const int *in, double *out, size_t n)
{
	for (size_t i = 0; i<n; i++)
	{
		out[i] = in[i] + 1;
	}
}