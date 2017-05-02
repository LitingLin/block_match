#include "test_common.h"
#include "intrinsic_func_helper.hpp"

BOOST_AUTO_TEST_CASE(test_sum8f)
{
	__m256 a = _mm256_set_ps(1, 2, 3, 4, 5, 6, 7, 8);
	BOOST_CHECK_EQUAL(sum8f_avx(a), 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}
BOOST_AUTO_TEST_CASE(test_sum4d)
{
	__m256d a = _mm256_set_pd(1, 2, 3, 4);
	BOOST_CHECK_EQUAL(sum4d_avx(a), 1 + 2 + 3 + 4);
}
BOOST_AUTO_TEST_CASE(test_sum4f)
{
	__m128 a = _mm_set_ps(1, 2, 3, 4);
	BOOST_CHECK_EQUAL(sum4f_sse2(a), 1 + 2 + 3 + 4);
}
BOOST_AUTO_TEST_CASE(test_sum2d)
{
	__m128d a = _mm_set_pd(1, 2);
	BOOST_CHECK_EQUAL(sum2d_sse2(a), 1 + 2);
}