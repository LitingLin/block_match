#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>
#include <spdlog/fmt/fmt.h>

#include "lib_match_internal.h"

memory_allocation_counter g_memory_allocator;

memory_allocation_counter::memory_allocation_counter()
	: max_memory_size(0), max_page_locked_memory_size(0), max_gpu_memory_size(0),
	current_memory_size(0), current_page_locked_memory_size(0), current_gpu_memory_size(0)
{
}

void memory_allocation_counter::register_allocator(size_t size, memory_allocation_type type)
{
	switch (type)
	{
	case memory_allocation_type::memory: 
		max_memory_size += size;
		break;
	case memory_allocation_type::page_locked:
		max_page_locked_memory_size += size;
		break;
	case memory_allocation_type::gpu:
		max_gpu_memory_size += size;
		break;
	default: ;
	}
}

void memory_allocation_counter::allocated(size_t size, memory_allocation_type type)
{
	switch (type)
	{
	case memory_allocation_type::memory: 
		current_memory_size += size;
		break;
	case memory_allocation_type::page_locked:
		current_page_locked_memory_size += size;
		break;
	case memory_allocation_type::gpu:
		current_gpu_memory_size += size;
		break;
	default: ;
	}
}

void memory_allocation_counter::released(size_t size, memory_allocation_type type)
{
	switch (type)
	{
	case memory_allocation_type::memory: 
		current_memory_size -= size;
		break;
	case memory_allocation_type::page_locked:
		current_page_locked_memory_size -= size;
		break;
	case memory_allocation_type::gpu:
		current_gpu_memory_size -= size;
		break;
	default: ;
	}
}

void memory_allocation_counter::trigger_error(size_t size, memory_allocation_type type) const
{
	std::string type_string;
	switch (type)
	{
	case memory_allocation_type::memory: 
		type_string = "System";
		break;
	case memory_allocation_type::page_locked:
		type_string = "Page locked";
		break;
	case memory_allocation_type::gpu:
		type_string = "GPU";
		break;
	default: ;
	}
	type_string = fmt::format(
		"{} memory allocation failed with {} bytes.\n"
		"\tCurrent\tMax(estimated)\n"
		"System:\t{}\t{}\n"
		"Page Locked:\t{}\t{}\n"
		"GPU:\t{}\t{}",
		type_string, size,
		current_memory_size, max_memory_size,
		current_page_locked_memory_size, max_page_locked_memory_size,
		current_gpu_memory_size, max_gpu_memory_size
	);

	switch (type)
	{
	case memory_allocation_type::memory:
		throw memory_alloc_exception(type_string);
		break;
	case memory_allocation_type::page_locked:
		throw page_locked_memory_alloc_exception(type_string);
		break;
	case memory_allocation_type::gpu:
		throw gpu_memory_alloc_exception(type_string);
		break;
	default:;
	}
}

void memory_allocation_counter::get_max_memory_required(size_t* max_memory_size_, size_t* max_page_locked_memory_size_, size_t* max_gpu_memory_size_)
{
	*max_memory_size_ = this->max_memory_size;
	*max_page_locked_memory_size_ = this->max_page_locked_memory_size;
	*max_gpu_memory_size_ = this->max_gpu_memory_size;
}

void diagnose::getMaxMemoryUsage(size_t* max_memory_size, size_t* max_page_locked_memory_size, size_t* max_gpu_memory_size)
{
	g_memory_allocator.get_max_memory_required(max_memory_size,
		max_page_locked_memory_size, max_gpu_memory_size);
}

memory_alloc_exception::memory_alloc_exception(const std::string& _Message,
	size_t max_memory_size, size_t max_page_locked_memory_size, size_t max_gpu_memory_size,
	size_t current_memory_size, size_t current_page_locked_memory_size, size_t current_gpu_memory_size)
	: runtime_error(_Message),
	max_memory_size(max_memory_size),
	max_page_locked_memory_size(max_page_locked_memory_size),
	max_gpu_memory_size(max_gpu_memory_size),
	current_memory_size(current_memory_size),
	current_page_locked_memory_size(current_page_locked_memory_size),
	current_gpu_memory_size(current_gpu_memory_size)
{
}

memory_alloc_exception::memory_alloc_exception(const char* _Message,
	size_t max_memory_size, size_t max_page_locked_memory_size, size_t max_gpu_memory_size,
	size_t current_memory_size, size_t current_page_locked_memory_size, size_t current_gpu_memory_size)
	: runtime_error(_Message),
	max_memory_size(max_memory_size),
	max_page_locked_memory_size(max_page_locked_memory_size),
	max_gpu_memory_size(max_gpu_memory_size),
	current_memory_size(current_memory_size),
	current_page_locked_memory_size(current_page_locked_memory_size),
	current_gpu_memory_size(current_gpu_memory_size)
{
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