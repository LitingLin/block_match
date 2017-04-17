#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>
#include <spdlog/fmt/fmt.h>

#include "lib_match_internal.h"

memory_allocation_counter g_memory_allocator;

malloc_type::malloc_type(values type)
	: type(type)
{
}

malloc_type::operator values() const
{
	return type;
}

malloc_type::operator std::string() const
{
	switch (type)
	{
	case values::memory:
		return "System";
		break;
	case values::page_locked:
		return "Page locked";
		break;
	case values::gpu:
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

void memory_allocation_counter::register_allocator(size_t size, malloc_type type)
{
	switch (malloc_type::values(type))
	{
	case malloc_type::values::memory:
		max_memory_size += size;
		break;
	case malloc_type::values::page_locked:
		max_page_locked_memory_size += size;
		break;
	case malloc_type::values::gpu:
		max_gpu_memory_size += size;
		break;
	default:;
	}
}

void memory_allocation_counter::allocated(size_t size, malloc_type type)
{
	switch (malloc_type::values(type))
	{
	case malloc_type::values::memory:
		current_memory_size += size;
		break;
	case malloc_type::values::page_locked:
		current_page_locked_memory_size += size;
		break;
	case malloc_type::values::gpu:
		current_gpu_memory_size += size;
		break;
	default:;
	}
}

void memory_allocation_counter::released(size_t size, malloc_type type)
{
	switch (malloc_type::values(type))
	{
	case malloc_type::values::memory:
		current_memory_size -= size;
		break;
	case malloc_type::values::page_locked:
		current_page_locked_memory_size -= size;
		break;
	case malloc_type::values::gpu:
		current_gpu_memory_size -= size;
		break;
	default:;
	}
}

void memory_allocation_counter::trigger_error(size_t size, malloc_type type) const
{
	throw memory_alloc_exception(fmt::format(
		"{} memory allocation failed with {} bytes.\n"
		"\tCurrent\tMax(estimated)\n"
		"System:\t{}\t{}\n"
		"Page Locked:\t{}\t{}\n"
		"GPU:\t{}\t{}",
		std::string(malloc_type(type)), size,
		current_memory_size, max_memory_size,
		current_page_locked_memory_size, max_page_locked_memory_size,
		current_gpu_memory_size, max_gpu_memory_size
	),
		type,
		max_memory_size, max_page_locked_memory_size, max_gpu_memory_size,
		current_memory_size, current_page_locked_memory_size, current_gpu_memory_size);
}

void memory_allocation_counter::get_max_memory_required(size_t* max_memory_size_, size_t* max_page_locked_memory_size_,
	size_t* max_gpu_memory_size_) const
{
	*max_memory_size_ = this->max_memory_size;
	*max_page_locked_memory_size_ = this->max_page_locked_memory_size;
	*max_gpu_memory_size_ = this->max_gpu_memory_size;
}

template <typename Type>
void BlockMatch<Type>::Diagnose::getMaxMemoryUsage(size_t* max_memory_size, size_t* max_page_locked_memory_size,
	size_t* max_gpu_memory_size)
{
	g_memory_allocator.get_max_memory_required(max_memory_size,
		max_page_locked_memory_size, max_gpu_memory_size);
}
LIB_MATCH_EXPORT
template 
void BlockMatch<float>::Diagnose::getMaxMemoryUsage(size_t*, size_t*, size_t*);
LIB_MATCH_EXPORT
template 
void BlockMatch<double>::Diagnose::getMaxMemoryUsage(size_t*, size_t* , size_t*);

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