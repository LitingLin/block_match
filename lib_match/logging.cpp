#include <spdlog/logger.h>

#include "lib_match_internal.h"

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

const char *get_base_file_name(const char *file_name)
{
	const char *base_file_name = file_name;
	while (*file_name != '\0') {
		if (*file_name == '\\' || *file_name == '/')
			base_file_name = file_name + 1;
		file_name++;
	}

	return base_file_name;
}

fatal_error_logging::fatal_error_logging(const char* file, int line, const char* function)
{
	str_stream << '[' << get_base_file_name(file) << ':' << line << ' ' << function << "] ";
}

fatal_error_logging::fatal_error_logging(const char* file, int line, const char* function, const char* exp): fatal_error_logging(file, line, function)
{
	str_stream << "Check failed: " << exp << ' ';
}

fatal_error_logging::fatal_error_logging(const char* file, int line, const char* function, const char* exp1, const char* op, const char* exp2): fatal_error_logging(file, line, function)
{
	str_stream << "Check failed: " << exp1 << ' ' << op << ' ' << exp2 << ' ';
}

fatal_error_logging::~fatal_error_logging() noexcept(false)
{
	str_stream << std::endl
		<< "*** Check failure stack trace: ***" << std::endl
		<< getStackTrace();
	if (std::uncaught_exception())
		logger.error("Fatal error occured during exception handling. Message: {}", str_stream.str());
	else
		throw std::runtime_error(str_stream.str());
}

std::ostringstream& fatal_error_logging::stream()
{
	return str_stream;
}

warning_logging::warning_logging(const char* file, int line, const char* function)
{
	str_stream << '[' << get_base_file_name(file) << ':' << line << ' ' << function << "] ";
}

warning_logging::warning_logging(const char* file, int line, const char* function, const char* exp) : warning_logging(file, line, function)
{
	str_stream << "Check failed: " << exp << ' ';
}

warning_logging::warning_logging(const char* file, int line, const char* function, const char* exp1, const char* op, const char* exp2) : warning_logging(file, line, function)
{
	str_stream << "Check failed: " << exp1 << ' ' << op << ' ' << exp2 << ' ';
}

warning_logging::~warning_logging()
{
	str_stream << std::endl
		<< "*** Check failure stack trace: ***" << std::endl
		<< getStackTrace();
	logger.warn(str_stream.str());
}

std::ostringstream& warning_logging::stream()
{
	return str_stream;
}
