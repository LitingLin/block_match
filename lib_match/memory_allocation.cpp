#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>
#include <spdlog/fmt/fmt.h>

#include "lib_match_internal.h"

memory_allocation_counter g_memory_allocator;

malloc_type::malloc_type(malloc_type_enum type)
	: type(type)
{
}

malloc_type::operator malloc_type_enum() const
{
	return type;
}

malloc_type::operator std::string() const
{
	switch (type)
	{
	case malloc_type_enum::memory:
		return "System";
		break;
	case malloc_type_enum::page_locked:
		return "Page locked";
		break;
	case malloc_type_enum::gpu:
		return "GPU";
		break;
	default:;
		NOT_IMPLEMENTED_ERROR;
		return "";
	}
}

memory_allocation_counter::memory_allocation_counter()
	: max_memory_size(0), max_page_locked_memory_size(0), max_gpu_memory_size(0),
	current_memory_size(0), current_page_locked_memory_size(0), current_gpu_memory_size(0)
{
}

void memory_allocation_counter::register_allocator(size_t size, malloc_type_enum type)
{
	switch (type)
	{
	case malloc_type_enum::memory:
		max_memory_size += size;
		break;
	case malloc_type_enum::page_locked:
		max_page_locked_memory_size += size;
		break;
	case malloc_type_enum::gpu:
		max_gpu_memory_size += size;
		break;
	default:;
	}
}

void memory_allocation_counter::allocated(size_t size, malloc_type_enum type)
{
	switch (type)
	{
	case malloc_type_enum::memory:
		current_memory_size += size;
		break;
	case malloc_type_enum::page_locked:
		current_page_locked_memory_size += size;
		break;
	case malloc_type_enum::gpu:
		current_gpu_memory_size += size;
		break;
	default:;
	}
}

void memory_allocation_counter::released(size_t size, malloc_type_enum type)
{
	switch (type)
	{
	case malloc_type_enum::memory:
		current_memory_size -= size;
		break;
	case malloc_type_enum::page_locked:
		current_page_locked_memory_size -= size;
		break;
	case malloc_type_enum::gpu:
		current_gpu_memory_size -= size;
		break;
	default:;
	}
}

void memory_allocation_counter::trigger_error(size_t size, malloc_type_enum type) const
{
	throw memory_alloc_exception(fmt::format(
		"{} memory allocation failed with {} bytes.\n"
		"\tCurrent\tMax(estimated)\n"
		"System:\t{}\t{}\n"
		"Page Locked:\t{}\t{}\n"
		"GPU:\t{}\t{}",
		type, size,
		current_memory_size, max_memory_size,
		current_page_locked_memory_size, max_page_locked_memory_size,
		current_gpu_memory_size, max_gpu_memory_size
	),
		type,
		max_memory_size, max_page_locked_memory_size, max_gpu_memory_size,
		current_memory_size, current_page_locked_memory_size, current_gpu_memory_size);
}

void memory_allocation_counter::get_max_memory_required(size_t* max_memory_size_, size_t* max_page_locked_memory_size_, size_t* max_gpu_memory_size_) const
{
	*max_memory_size_ = this->max_memory_size;
	*max_page_locked_memory_size_ = this->max_page_locked_memory_size;
	*max_gpu_memory_size_ = this->max_gpu_memory_size;
}

template <typename Type>
void BlockMatch<Type>::Diagnose::getMaxMemoryUsage(size_t* max_memory_size, size_t* max_page_locked_memory_size, size_t* max_gpu_memory_size)
{
	g_memory_allocator.get_max_memory_required(max_memory_size,
		max_page_locked_memory_size, max_gpu_memory_size);
}
LIB_MATCH_EXPORT
template 
void BlockMatch<float>::Diagnose::getMaxMemoryUsage(size_t* max_memory_size, size_t* max_page_locked_memory_size, size_t* max_gpu_memory_size);
LIB_MATCH_EXPORT
template 
void BlockMatch<double>::Diagnose::getMaxMemoryUsage(size_t* max_memory_size, size_t* max_page_locked_memory_size, size_t* max_gpu_memory_size);

memory_alloc_exception::memory_alloc_exception(const std::string& _Message,
	malloc_type type,
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
	malloc_type type,
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

malloc_type memory_alloc_exception::get_memory_allocation_type() const
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