#pragma once

#ifdef __unix__
#include <pthread.h>
#include <semaphore.h>
#include <tbb/concurrent_queue.h>
#endif
#include <string>

class execution_service
{
public:
	enum class task_state
	{
		NEW,
		PROCESSING,
		DONE
	};

	execution_service(unsigned n_threads = 4);

	execution_service(const execution_service&) = delete;

	~execution_service();
	
	void* submit(unsigned int(*func)(void *), void* para);

	void join(void* task_handle) const;

	task_state query(void* task_handle) const;

	void release(void* task_handle) const;

	unsigned int get_rc(void* task_handle) const;

	std::string &get_exp_what(void* task_handle) const;
private:
#ifdef _MSC_VER
	void *pool;
#elif defined __unix__
	static void *start_routine(void *para);
	unsigned int m_size;
	std::atomic<bool> m_exit_flag;
	struct _task;
	tbb::concurrent_queue<_task*> m_task_queue;
	tbb::concurrent_queue<unsigned int> m_free_thread_queue;
	pthread_t *m_threads;
	sem_t *m_sems;
#endif
};
