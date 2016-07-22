#include "test_common.h"

bool validate_pad_result(float *to_validate, float *truth_array, int src_m, int src_n, int truth_m, int truth_n)
{
	
}

BOOST_AUTO_TEST_CASE(test_padding)
{
	float to_pad[] =
	{ 1,2,3,
		4,5,6,
		7,8,9 };
	float padded[] =
	{ 9,8,7,7,8,9,9,8,7,
	6,5,4,4,5,6,6,5,4,
	3,2,1,1,2,3,3,2,1,
	3,2,1,1,2,3,3,2,1,
	6,5,4,4,5,6,6,5,4,
	9,8,7,7,8,9,9,8,7,
	9,8,7,7,8,9,9,8,7,
	6,5,4,4,5,6,6,5,4,
	3,2,1,1,2,3,3,2,1 };

	float buf[32];
	copyBlockWithSymmetricPaddding(buf, to_pad, 3, 3, );
}