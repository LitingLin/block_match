#pragma once

#include <atomic>
#include <functional>
#include <concurrent_queue.h>

typedef void* HANDLE;
class thread_pool
{
public:
	enum class task_state
	{
		NEW,
		PROCESSING,
		DONE
	};
private:
	struct _task;
public:
	thread_pool(unsigned int num);

	thread_pool(const thread_pool&) = delete;

	~thread_pool();

	void* submit(std::function<unsigned int(void*)> task, void* para);

	void join(void* task_handle) const;

	task_state query(void* task_handle) const;

	void release(void* task_handle);

	unsigned int get_rc(void* task_handle);
private:
	unsigned static int __stdcall thread_helper(void* para);

	unsigned int new_tid();
	std::atomic<bool> m_exit_flag;
	Concurrency::concurrent_queue<_task*> m_task_queue;
	Concurrency::concurrent_queue<unsigned int>m_free_thread_queue;
	HANDLE *m_wait_event;
	HANDLE *m_hThreads;
	std::atomic<unsigned int> m_max_task_id;
	unsigned int m_size;
};