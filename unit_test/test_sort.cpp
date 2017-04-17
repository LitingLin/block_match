#include "test_common.h"

BOOST_AUTO_TEST_CASE( test_sort )
{
	float vals[] = { 3,2,5,7,4 };
	int index[] = { 0,1,2,3,4 };
	const unsigned vals_size = sizeof(vals) / sizeof(*vals);
	const unsigned index_size = sizeof(index) / sizeof(*index);
	static_assert(vals_size == index_size, "size mismatch");
	lib_match_sort(index, vals, sizeof(vals) / sizeof(*vals));
	int truth_index[] = { 1,0,4,2,3 };
	BOOST_CHECK_EQUAL_COLLECTIONS(index, index + index_size, truth_index, truth_index + index_size);
}