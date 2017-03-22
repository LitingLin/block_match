#include "thread_pool.h"

#ifdef _MSC_VER

#define VC_EXTRALEAN
#include <Windows.h>
#include <cstdlib>

execution_service::execution_service(unsigned num)
	: pool(CreateThreadpool(nullptr))
{
#ifndef NDEBUG
	if (!pool)
		abort();
#endif

	SetThreadpoolThreadMaximum((PTP_POOL)pool, num);
	BOOL isOK = SetThreadpoolThreadMinimum((PTP_POOL)pool, 0);
#ifndef NDEBUG
	if (!isOK)
		abort();
#endif
}

execution_service::~execution_service()
{
	CloseThreadpool((PTP_POOL)pool);
}

struct work_context
{
	PTP_WORK work;
	unsigned(*func)(void*);
	void *para;
	unsigned int rc;
	std::string message;
	execution_service::task_state state;
};

void __stdcall start_routine(
	PTP_CALLBACK_INSTANCE Instance,
	PVOID                 Context,
	PTP_WORK              Work
	)
{
	work_context *work_context = static_cast<struct work_context*>(Context);
	work_context->state = execution_service::task_state::PROCESSING;
	work_context->rc = 1;
	try {
		work_context->rc = work_context->func(work_context->para);
	}
	catch (std::exception &exp){
		work_context->message = exp.what();
	}
	work_context->state = execution_service::task_state::DONE;
}

void* execution_service::submit(unsigned(* func)(void*), void* para)
{
	struct work_context *work_context = new ::work_context;
	work_context->func = func; work_context->para = para;
	work_context->work = CreateThreadpoolWork(start_routine, work_context, nullptr);
	SubmitThreadpoolWork(work_context->work);
	return work_context;
}

void execution_service::join(void* task_handle) const
{
	struct work_context * work_context = static_cast<struct work_context *>(task_handle);
	WaitForThreadpoolWorkCallbacks(work_context->work, FALSE);
}

execution_service::task_state execution_service::query(void* task_handle) const
{
	struct work_context * work_context = static_cast<struct work_context *>(task_handle);
	return work_context->state;
}

void execution_service::release(void* task_handle) const
{
	struct work_context * work_context = static_cast<struct work_context *>(task_handle);
	CloseThreadpoolWork(work_context->work);
	delete work_context;
}

unsigned execution_service::get_rc(void* task_handle) const
{
	struct work_context * work_context = static_cast<struct work_context *>(task_handle);
	return work_context->rc;
}

std::string& execution_service::get_exp_what(void* task_handle) const
{
	struct work_context * work_context = static_cast<struct work_context *>(task_handle);
	return work_context->message;
}

#elif defined __unix__

#ifndef NDEBUG
#include <string>
#include <stdexcept>
#endif

#ifndef NDEBUG
#define THREAD_POOL_CHECK(x) \
        if (x) throw std::runtime_error(std::to_string(errno));
#else
#define THREAD_POOL_CHECK(x) x
#endif

struct execution_service::_task
{
	std::atomic<task_state> state;
	unsigned int(*func)(void *);
	void *para;
	unsigned int rc;
	std::string message;
	std::atomic<sem_t *> sem;
};

struct thread_info
{
	execution_service *class_ptr;
	unsigned int thread_id;
};

execution_service::execution_service(unsigned num)
	: m_exit_flag(false), m_size(num)
{
	m_threads = new pthread_t[m_size];
	m_sems = new sem_t[m_size];

	for (unsigned int i = 0; i != m_size; i++)
	{
		THREAD_POOL_CHECK(sem_init(m_sems + i, 0, 0));
		THREAD_POOL_CHECK(pthread_create(m_threads + i, nullptr, start_routine, new thread_info{ this, i }));
	}
}

execution_service::~execution_service()
{
	m_exit_flag = true;

	for (unsigned int i = 0; i != m_size; i++)
		THREAD_POOL_CHECK(sem_post(m_sems + i));

	for (unsigned int i = 0; i != m_size; ++i)
		THREAD_POOL_CHECK(pthread_join(m_threads[i], nullptr));

	for (unsigned int i = 0; i != m_size; ++i) {
		THREAD_POOL_CHECK(sem_destroy(m_sems + i));
	}

	delete[] m_threads;
	delete[] m_sems;
}

void* execution_service::submit(unsigned int(*func)(void *), void* para)
{
	_task *task_entity = new _task;
	task_entity->func = func;
	task_entity->state = task_state::NEW;
	task_entity->sem.store(nullptr);
	task_entity->para = para;

	m_task_queue.push(task_entity);

	unsigned int free_thread_id;
	if (m_free_thread_queue.try_pop(free_thread_id))
		THREAD_POOL_CHECK(sem_post(m_sems + free_thread_id));

	return task_entity;
}

void execution_service::join(void* task_handle) const
{
	_task *task_entity = static_cast<_task*>(task_handle);

	if (task_entity->state.load() == task_state::DONE)
		return;
	sem_t *sem = task_entity->sem.load();
	if (sem == nullptr)
	{
		sem = (sem_t*)malloc(sizeof(sem_t));
		THREAD_POOL_CHECK(sem_init(sem, 0, 0));
		sem_t *expect = 0;
		if (!task_entity->sem.compare_exchange_strong(expect, sem)) {
			THREAD_POOL_CHECK(sem_destroy(sem));
			free(sem);
			if (task_entity->state.load() == task_state::DONE)
				task_entity->sem = nullptr;
			return;
		}
		THREAD_POOL_CHECK(sem_wait(sem));
	}
}

execution_service::task_state execution_service::query(void* task_handle) const
{
	return static_cast<_task*>(task_handle)->state;
}

void execution_service::release(void* task_handle) const
{
	_task *task_entity = static_cast<_task*>(task_handle);
	sem_t *sem = task_entity->sem;
	if (sem && sem != (sem_t*)1) {
		THREAD_POOL_CHECK(sem_destroy(sem));
		free(sem);
	}

	delete task_entity;
}

unsigned execution_service::get_rc(void* task_handle) const
{
	return static_cast<_task*>(task_handle)->rc;
}

void *execution_service::start_routine(void *para)
{
	struct thread_info *thread_info = (struct thread_info *)para;
	execution_service *this_class = thread_info->class_ptr;
	unsigned int tid = thread_info->thread_id;
	delete thread_info;

	tbb::concurrent_queue<_task*> &task_queue = this_class->m_task_queue;
	tbb::concurrent_queue<unsigned int> &free_thread_queue = this_class->m_free_thread_queue;
	std::atomic<bool> &exit_flag = this_class->m_exit_flag;

	sem_t *sem = this_class->m_sems + tid;

	for (;;)
	{
		_task *task_entity;

		if (!task_queue.try_pop(task_entity))
		{
			if (exit_flag)
				break;

			free_thread_queue.push(tid);

			THREAD_POOL_CHECK(sem_wait(sem));

			continue;
		}

		task_entity->state = task_state::PROCESSING;

		try {
			task_entity->rc = task_entity->func(task_entity->para);
		}
		catch (std::exception &exp) {
			task_entity->message = exp.what();
		}

		sem_t *task_sem = task_entity->sem;
		if (!task_entity->sem.compare_exchange_strong(task_sem, (sem_t*)1))
		{
			task_sem = task_entity->sem;
		}
		if (task_sem) {
			THREAD_POOL_CHECK(sem_post(task_sem));
		}
		task_entity->state = task_state::DONE;
	}

	return 0;
}

std::string& execution_service::get_exp_what(void* task_handle) const
{
	_task *task_entity = static_cast<_task*>(task_handle);
	return task_entity->message;
}

#endif