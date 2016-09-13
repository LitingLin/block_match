#include <spdlog/logger.h>

#include "lib_match.h"

LibMatchSinkFunction *sinkFunction = nullptr;

void libMatchRegisterLoggingSinkFunction(LibMatchSinkFunction sinkFunction)
{
	::sinkFunction = sinkFunction;
}

class custom_sink : public spdlog::sinks::base_sink<std::mutex>
{
public:
	void flush() override;
protected:
	void _sink_it(const spdlog::details::log_msg& msg) override;
};

void custom_sink::flush()
{

}
void custom_sink::_sink_it(const spdlog::details::log_msg& msg)
{
	if (sinkFunction != nullptr)
		sinkFunction(msg.formatted.c_str());
}

spdlog::logger logger("logging", std::make_shared<custom_sink>());