#include <spdlog/logger.h>

#include "block_match.h"

SinkFunction *sink_function = nullptr;

extern "C"
void registerLoggingSinkFunction(SinkFunction sinkFunction)
{
	sink_function = sinkFunction;
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
	if (sink_function != nullptr)
		sink_function(msg.formatted.c_str());
}

spdlog::logger logger("logging", std::make_shared<custom_sink>());