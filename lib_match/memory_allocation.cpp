#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>
#include <spdlog/fmt/fmt.h>

#include "lib_match_internal.h"

memory_allocation_statistic g_memory_statistic;

std::string to_string(memory_type type)
{
	switch (type)
	{
	case memory_type::system:
		return "System";
		break;
	case memory_type::page_locked:
		return "Page locked";
		break;
	case memory_type::gpu:
		return "GPU";
		break;
	default:;
		NOT_IMPLEMENTED_ERROR;
		return "";
	}
}

memory_allocation_statistic::memory_allocation_statistic()
	: max_memory_size(0), max_page_locked_memory_size(0), max_gpu_memory_size(0),
	current_memory_size(0), current_page_locked_memory_size(0), current_gpu_memory_size(0)
{
}

void memory_allocation_statistic::register_allocator(size_t size, memory_type type)
{
	switch (type)
	{
	case memory_type::system:
		max_memory_size += size;
		break;
	case memory_type::page_locked:
		max_page_locked_memory_size += size;
		break;
	case memory_type::gpu:
		max_gpu_memory_size += size;
		break;
	default:;
	}
}

void memory_allocation_statistic::unregister_allocator(size_t size, memory_type type)
{
	switch (type)
	{
	case memory_type::system:
		max_memory_size -= size;
		break;
	case memory_type::page_locked:
		max_page_locked_memory_size -= size;
		break;
	case memory_type::gpu:
		max_gpu_memory_size -= size;
		break;
	default:;
	}
}

void memory_allocation_statistic::allocated(size_t size, memory_type type)
{
	switch (type)
	{
	case memory_type::system:
		current_memory_size += size;
		break;
	case memory_type::page_locked:
		current_page_locked_memory_size += size;
		break;
	case memory_type::gpu:
		current_gpu_memory_size += size;
		break;
	default:;
	}
}

void memory_allocation_statistic::released(size_t size, memory_type type)
{
	switch (type)
	{
	case memory_type::system:
		current_memory_size -= size;
		break;
	case memory_type::page_locked:
		current_page_locked_memory_size -= size;
		break;
	case memory_type::gpu:
		current_gpu_memory_size -= size;
		break;
	default:;
	}
}

void memory_allocation_statistic::trigger_error(size_t size, memory_type type) const
{
	throw memory_alloc_exception(fmt::format(
		"{} system allocation failed with {} bytes.\n"
		"\tCurrent\tMax(estimated)\n"
		"System:\t{}\t{}\n"
		"Page Locked:\t{}\t{}\n"
		"GPU:\t{}\t{}\n",
		to_string(type), size,
		current_memory_size, max_memory_size,
		current_page_locked_memory_size, max_page_locked_memory_size,
		current_gpu_memory_size, max_gpu_memory_size
	),
		type,
		max_memory_size, max_page_locked_memory_size, max_gpu_memory_size,
		current_memory_size, current_page_locked_memory_size, current_gpu_memory_size);
}

void memory_allocation_statistic::get_max_memory_required(size_t* max_memory_size_, size_t* max_page_locked_memory_size_,
	size_t* max_gpu_memory_size_) const
{
	*max_memory_size_ = this->max_memory_size;
	*max_page_locked_memory_size_ = this->max_page_locked_memory_size;
	*max_gpu_memory_size_ = this->max_gpu_memory_size;
}

void memory_allocation_statistic::get_current_memory_usage(size_t *current_memory_size_,
	size_t *current_page_locked_memory_size_, size_t *current_gpu_memory_size_) const
{
	*current_memory_size_ = this->current_memory_size;
	*current_page_locked_memory_size_ = this->current_page_locked_memory_size;
	*current_gpu_memory_size_ = this->current_gpu_memory_size;
}

void LibMatchDiagnose::getMaxMemoryUsage(size_t* max_memory_size, size_t* max_page_locked_memory_size,
	size_t* max_gpu_memory_size)
{
	g_memory_statistic.get_max_memory_required(max_memory_size,
		max_page_locked_memory_size, max_gpu_memory_size);
}

void LibMatchDiagnose::getCurrentMemoryUsage(size_t *current_memory_size,
	size_t *current_page_locked_memory_size, size_t *current_gpu_memory_size)
{
	g_memory_statistic.get_current_memory_usage(current_memory_size,
		current_page_locked_memory_size, current_gpu_memory_size);
}

memory_alloc_exception::memory_alloc_exception(const std::string& _Message,
	memory_type type,
	size_t max_memory_size, size_t max_page_locked_memory_size, size_t max_gpu_memory_size,
	size_t current_memory_size, size_t current_page_locked_memory_size, size_t current_gpu_memory_size)
	: runtime_error(_Message),
	type(type),
	max_memory_size(max_memory_size),
	max_page_locked_memory_size(max_page_locked_memory_size),
	max_gpu_memory_size(max_gpu_memory_size),
	current_memory_size(current_memory_size),
	current_page_locked_memory_size(current_page_locked_memory_size),
	current_gpu_memory_size(current_gpu_memory_size)
{
}

memory_alloc_exception::memory_alloc_exception(const char* _Message,
	memory_type type,
	size_t max_memory_size, size_t max_page_locked_memory_size, size_t max_gpu_memory_size,
	size_t current_memory_size, size_t current_page_locked_memory_size, size_t current_gpu_memory_size)
	: runtime_error(_Message),
	type(type),
	max_memory_size(max_memory_size),
	max_page_locked_memory_size(max_page_locked_memory_size),
	max_gpu_memory_size(max_gpu_memory_size),
	current_memory_size(current_memory_size),
	current_page_locked_memory_size(current_page_locked_memory_size),
	current_gpu_memory_size(current_gpu_memory_size)
{
}

memory_type memory_alloc_exception::get_memory_allocation_type() const
{
	return type;
}

size_t memory_alloc_exception::get_max_memory_size() const
{
	return max_memory_size;
}

size_t memory_alloc_exception::get_max_page_locked_memory_size() const
{
	return max_page_locked_memory_size;
}

size_t memory_alloc_exception::get_max_gpu_memory_size() const
{
	return max_gpu_memory_size;
}

size_t memory_alloc_exception::get_current_memory_size() const
{
	return current_memory_size;
}

size_t memory_alloc_exception::get_current_page_locked_memory_size() const
{
	return current_page_locked_memory_size;
}

size_t memory_alloc_exception::get_current_gpu_memory_size() const
{
	return current_gpu_memory_size;
}

void *aligned_block_malloc(size_t size, size_t alignment)
{
	if (size % alignment)
		size += (alignment - size % alignment);

	void *ptr = _aligned_malloc(size, alignment);

	if (!ptr)
		throw std::runtime_error("");

	memset(ptr, 0, size);
	return ptr;
}

void aligned_free(void *ptr)
{
	_aligned_free(ptr);
}