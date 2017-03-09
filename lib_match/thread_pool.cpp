#include "thread_pool.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <process.h>

struct multi_task_service::_task
{
	std::atomic<task_state> state;
	unsigned int(*func)(void *);
	void *para;
	unsigned int rc;
	std::atomic<HANDLE> hEvent;
};

multi_task_service::multi_task_service(unsigned num)
	: m_exit_flag(false), m_max_task_id(0), m_size(num)
{
	m_hThreads = new HANDLE[m_size];
	m_wait_event = new HANDLE[m_size];

	for (unsigned int i = 0; i != m_size; i++)
	{
		uintptr_t handle = _beginthreadex(nullptr, 0, start_routine, this, 0, nullptr);
		m_hThreads[i] = reinterpret_cast<HANDLE>(handle);
		m_wait_event[i] = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	}
}

multi_task_service::~multi_task_service()
{
	if (!m_exit_flag)
		shutdown();
}

void multi_task_service::shutdown()
{
	m_exit_flag = true;

	for (unsigned int i = 0; i != m_size; i++)
		SetEvent(m_wait_event[i]);

	if (m_size > MAXIMUM_WAIT_OBJECTS)
	{
		unsigned int i = 0;
		for (; i <= m_size - MAXIMUM_WAIT_OBJECTS; i += MAXIMUM_WAIT_OBJECTS)
			WaitForMultipleObjectsEx(MAXIMUM_WAIT_OBJECTS, m_hThreads + i, TRUE, INFINITE, FALSE);
		if (i != m_size)
			WaitForMultipleObjectsEx(m_size - i, m_hThreads + i, TRUE, INFINITE, FALSE);
	}
	else
	{
		WaitForMultipleObjectsEx(m_size, m_hThreads, TRUE, INFINITE, FALSE);
	}

	for (unsigned int i = 0; i != m_size; i++)
		CloseHandle(m_wait_event[i]);

	delete[]m_hThreads;
	delete[]m_wait_event;
}

void* multi_task_service::submit(unsigned int(*func)(void *), void* para)
{
	_task *task_entity = new _task;
	task_entity->func = func;
	task_entity->state = task_state::NEW;
	task_entity->hEvent = nullptr;
	task_entity->para = para;

	m_task_queue.push(task_entity);

	unsigned int free_thread_id;
	if (m_free_thread_queue.try_pop(free_thread_id))
		SetEvent(m_wait_event[free_thread_id]);

	return task_entity;
}

bool multi_task_service::join(void* task_handle, uint32_t timeout) const
{
	_task *task_entity = static_cast<_task*>(task_handle);

	if (task_entity->state == task_state::DONE)
		return true;

	if (!task_entity->hEvent)
	{
		HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

		HANDLE expect = nullptr;
		if (!task_entity->hEvent.compare_exchange_strong(expect, hEvent) || task_entity->state == task_state::DONE) {
			CloseHandle(hEvent);
			if (task_entity->state == task_state::DONE)
				task_entity->hEvent = nullptr;
			return true;
		}
	}

	if (WaitForSingleObjectEx(task_entity->hEvent, timeout, FALSE) == WAIT_TIMEOUT)
		return false;
	else
		return true;
}

multi_task_service::task_state multi_task_service::query(void* task_handle) const
{
	return static_cast<_task*>(task_handle)->state;
}

void multi_task_service::release(void* task_handle) const
{
	_task *task_entity = static_cast<_task*>(task_handle);
	HANDLE hEvent = task_entity->hEvent;
	if (hEvent)
		CloseHandle(hEvent);

	delete task_handle;
}

void multi_task_service::kill(void* task_handle)
{

}

unsigned multi_task_service::get_rc(void* task_handle)
{
	return static_cast<_task*>(task_handle)->rc;
}

unsigned multi_task_service::start_routine(void* para)
{
	multi_task_service *this_class = static_cast<multi_task_service*>(para);
	Concurrency::concurrent_queue<_task*> &task_queue = this_class->m_task_queue;
	Concurrency::concurrent_queue<unsigned int> &free_thread_queue = this_class->m_free_thread_queue;
	std::atomic<bool> &exit_flag = this_class->m_exit_flag;

	unsigned int tid = this_class->new_tid();

	HANDLE hEvnet = this_class->m_wait_event[tid];


	for (;;)
	{
		_task *task_entity;

		if (!task_queue.try_pop(task_entity))
		{
			if (exit_flag)
				break;

			free_thread_queue.push(tid);

			WaitForSingleObjectEx(hEvnet, INFINITE, FALSE);

			continue;
		}

		task_entity->state = task_state::PROCESSING;
		task_entity->rc = task_entity->func(task_entity->para);
		task_entity->state = task_state::DONE;

		HANDLE waiting_thread_handle = task_entity->hEvent;
		if (!task_entity->hEvent.compare_exchange_strong(waiting_thread_handle, INVALID_HANDLE_VALUE))
			waiting_thread_handle = task_entity->hEvent;
		if (waiting_thread_handle)
			SetEvent(waiting_thread_handle);
	}

	return 0;
}

unsigned multi_task_service::new_tid()
{
	unsigned int task_id = m_max_task_id.fetch_add(1);

	return task_id;
}