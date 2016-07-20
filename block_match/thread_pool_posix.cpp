#ifndef _MSC_VER
/*
 * Author: Liting Lin
 */

#include "thread_pool.h"

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

struct ThreadPool::_task
{
	std::atomic<task_state> state;
	unsigned int(*func)(void *);
	void *para;
	unsigned int rc;
	std::atomic<pthread_cond_t *> cond;
	pthread_mutex_t *mutex;
};

ThreadPool::ThreadPool(unsigned num)
	: m_exit_flag(false), m_max_task_id(0), m_size(num)
{
	m_threads = new pthread_t[m_size];
    m_condition_variables = new pthread_cond_t[m_size];
	m_mutexes = new pthread_mutex_t[m_size];

	for (unsigned int i = 0; i != m_size; i++)
	{
        THREAD_POOL_CHECK(pthread_create(m_threads + i, nullptr, start_routine, this));
        THREAD_POOL_CHECK(pthread_cond_init(m_condition_variables + i, nullptr));
		THREAD_POOL_CHECK(pthread_mutex_init(m_mutexes + i, nullptr));
	}
}

ThreadPool::~ThreadPool()
{
	m_exit_flag = true;

	for (unsigned int i = 0; i != m_size; i++)
        THREAD_POOL_CHECK(pthread_cond_signal(m_condition_variables + i));

    for (unsigned int i = 0;i != m_size; ++i)
        THREAD_POOL_CHECK(pthread_join(m_threads[i], nullptr));

    for (unsigned int i = 0;i != m_size; ++i) {
		THREAD_POOL_CHECK(pthread_cond_destroy(m_condition_variables + i));
		THREAD_POOL_CHECK(pthread_mutex_destroy(m_mutexes + i));
	}

	delete [] m_threads;
	delete [] m_condition_variables;
	delete [] m_mutexes;
}

void* ThreadPool::submit(unsigned int(*func)(void *), void* para)
{
	_task *task_entity = new _task;
	task_entity->func = func;
	task_entity->state = task_state::NEW;
	task_entity->cond.store(nullptr);
	task_entity->para = para;

	m_task_queue.push(task_entity);

	unsigned int free_thread_id;
	if (m_free_thread_queue.try_pop(free_thread_id))
		THREAD_POOL_CHECK(pthread_cond_signal(m_condition_variables + free_thread_id));

	return task_entity;
}

void ThreadPool::join(void* task_handle) const
{
	_task *task_entity = static_cast<_task*>(task_handle);

	if (task_entity->state.load() == task_state::DONE)
		return;
    pthread_cond_t *cond = task_entity->cond.load();
	pthread_mutex_t *mutex;
	if (cond == nullptr)
	{
        cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
		mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
        THREAD_POOL_CHECK(pthread_cond_init(cond, nullptr));
		THREAD_POOL_CHECK(pthread_mutex_init(mutex, nullptr));
		task_entity->mutex = mutex;
        pthread_cond_t *expect = 0;
		if (!task_entity->cond.compare_exchange_strong(expect, cond) || task_entity->state.load() == task_state::DONE) {
			THREAD_POOL_CHECK(pthread_cond_destroy(cond));
			THREAD_POOL_CHECK(pthread_mutex_destroy(mutex));
            free(cond);
			free(mutex);
			if (task_entity->state.load() == task_state::DONE)
				task_entity->cond = nullptr;
			return;
		}
	}
	else
		mutex = task_entity->mutex;

    THREAD_POOL_CHECK(pthread_cond_wait(cond, mutex));
}

ThreadPool::task_state thread_pool::query(void* task_handle) const
{
	return static_cast<_task*>(task_handle)->state;
}

void ThreadPool::release(void* task_handle)
{
	_task *task_entity = static_cast<_task*>(task_handle);
	pthread_cond_t *cond = task_entity->cond;
	if (cond) {
        THREAD_POOL_CHECK(pthread_cond_destroy(cond));
		THREAD_POOL_CHECK(pthread_mutex_destroy(task_entity->mutex));
        free(cond);
		free(task_entity->mutex);
    }

	delete task_entity;
}

unsigned ThreadPool::get_rc(void* task_handle)
{
	return static_cast<_task*>(task_handle)->rc;
}

void *ThreadPool::start_routine(void *para)
{
    thread_pool *this_class = static_cast<thread_pool*>(para);
    tbb::concurrent_queue<_task*> &task_queue = this_class->m_task_queue;
    tbb::concurrent_queue<unsigned int> &free_thread_queue = this_class->m_free_thread_queue;
    std::atomic<bool> &exit_flag = this_class->m_exit_flag;

    unsigned int tid = this_class->new_tid();

    pthread_cond_t condition_variable = this_class->m_condition_variables[tid];
	pthread_mutex_t mutex = this_class->m_mutexes[tid];

    for (;;)
    {
        _task *task_entity;

        if (!task_queue.try_pop(task_entity))
        {
            if (exit_flag)
                break;

            free_thread_queue.push(tid);

            THREAD_POOL_CHECK(pthread_cond_wait(&condition_variable, &mutex));

            continue;
        }

        task_entity->state = task_state::PROCESSING;
        unsigned int rc = task_entity->func(task_entity->para);
        task_entity->rc = rc;
        task_entity->state = task_state::DONE;

        pthread_cond_t *task_cond = task_entity->cond;
        if (!task_entity->cond.compare_exchange_strong(task_cond, (pthread_cond_t*)1))
		{
            task_cond = task_entity->cond;
		}
        if (task_cond) {
            THREAD_POOL_CHECK(pthread_cond_signal(task_cond));
		}
    }

    return 0;
}

unsigned ThreadPool::new_tid()
{
	unsigned int task_id = m_max_task_id.fetch_add(1);

	return task_id;
}
#endif
