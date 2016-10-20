#define BOOST_TEST_MODULE block_match
#include <boost/test/unit_test.hpp>

#include <lib_match.h>

void logging_sink(const char *msg)
{
	puts(msg);
}

class obj
{
public:
	obj()
	{
		libMatchRegisterLoggingSinkFunction(logging_sink);		
	}
}_obj;

char *getLastErrorString()
{
	static char buffer[LIB_MATCH_MAX_MESSAGE_LENGTH];
	libMatchGetLastErrorString(buffer, LIB_MATCH_MAX_MESSAGE_LENGTH);
	return buffer;
}