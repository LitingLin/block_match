#pragma once

#if defined _MSC_VER
#define FORCE_INLINE __forceinline
#elif defined __GNUC__
#define FORCE_INLINE __inline__ __attribute__((always_inline))
#endif

class thread_pool;

namespace internal{
	template<typename R, template<typename...> class Params, typename... Args, std::size_t... I> constexpr
		unsigned FORCE_INLINE thread_pool_base_function_helper(R(*func)(Args...), Params<Args...> const&params, std::index_sequence<I...>)
	{
		return func(std::get<I>(params)...);
	}

	template<typename FuncType, FuncType func, template<typename...> class Params, typename... Args>
	unsigned thread_pool_base_function(void *arg) {
		Params<Args...> const &params = *static_cast<Params<Args...> *>(arg);
		return thread_pool_base_function_helper(func, params, std::index_sequence_for<Args...>{});
	}
	template <typename Fn_Type, Fn_Type func, template<typename...> class Params, typename... Args>
	void* thread_pool_launcher_helper(thread_pool &pool, Params<Args...> & params)
	{
		return pool.submit(internal::thread_pool_base_function< Fn_Type, func, Params, Args... >, &params);
	}
}

#define thread_pool_launcher(pool, func, params) internal::thread_pool_launcher_helper<decltype(func), func>(pool, params);
