#include "test_common.h"

BOOST_AUTO_TEST_CASE(test_iterator) {
	void *ptr = (void *)0x18;
	ContiguousMemoryIterator iterator(ptr, 2);
	auto iterator2 = iterator.clone(10);
	BOOST_CHECK_EQUAL(iterator2->get(), (void*)(0x18 + 2 * 10));
}