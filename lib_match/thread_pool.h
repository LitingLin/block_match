#pragma once

/*
 * Author: Liting Lin
 */

#include <atomic>
#ifdef _MSC_VER
#include <concurrent_queue.h>
#else
#include <pthread.h>
#include <tbb/concurrent_queue.h>
#endif

typedef void* HANDLE;
class multi_task_service
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
	multi_task_service(unsigned num);

	multi_task_service(const multi_task_service&) = delete;

	~multi_task_service();

	void shutdown();

	void* submit(unsigned int(*func)(void *), void* para);

	void join(void* task_handle) const;

	task_state query(void* task_handle) const;

	void release(void* task_handle) const;

	unsigned int get_rc(void* task_handle);
private:
#ifdef _MSC_VER
	unsigned static int __stdcall start_routine(void* para);
#else
    static void *start_routine(void *para);
#endif

	unsigned int new_tid();
	std::atomic<bool> m_exit_flag;
#ifdef _MSC_VER
	Concurrency::concurrent_queue<_task*> m_task_queue;
	Concurrency::concurrent_queue<unsigned int> m_free_thread_queue;
	HANDLE *m_wait_event;
	HANDLE *m_hThreads;
#else
    tbb::concurrent_queue<_task*> m_task_queue;
    tbb::concurrent_queue<unsigned int> m_free_thread_queue;
    pthread_cond_t *m_condition_variables;
    pthread_t *m_threads;
    pthread_mutex_t *m_mutexes;
#endif
	std::atomic<unsigned int> m_max_task_id;
	unsigned int m_size;
};