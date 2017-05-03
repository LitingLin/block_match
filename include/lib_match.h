#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <bitset>
#include <array>
#include <typeindex>
#include <memory>

#if defined _WIN32
#ifdef LIB_MATCH_BUILD_DLL
#define LIB_MATCH_EXPORT 
#else
#define LIB_MATCH_EXPORT 
#endif
#else
#ifdef LIB_MATCH_BUILD_DLL
#define LIB_MATCH_EXPORT __attribute__ ((visibility ("default")))
#else
#define LIB_MATCH_EXPORT
#endif
#endif

enum class MeasureMethod { mse, cc };

enum class LibMatchErrorCode
{
	errorMemoryAllocation,
	errorPageLockedMemoryAllocation,
	errorGpuMemoryAllocation,
	errorCuda,
	errorInternal,
	success
};

enum class SearchType
{
	local,
	global
};

enum class PadMethod
{
	zero,
	circular,
	replicate,
	symmetric
};

enum class BorderType
{
	normal,
	includeLastBlock
};

enum class SearchFrom
{
	topLeft,
	center
};
/*
LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchInitialize(void **instance,
	int numberOfArrayA, int numberOfArrayB, int lengthOfArray);

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchExecute(void *instance, float *A, float *B, MeasureMethod method,
	float **result);

LIB_MATCH_EXPORT
LibMatchErrorCode arrayMatchFinalize(void *instance);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumMemoryAllocationSize();

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumGpuMemoryAllocationSize(int lengthOfArray);

LIB_MATCH_EXPORT
size_t arrayMatchGetMaximumPageLockedMemoryAllocationSize(int numberOfArrayA, int numberOfArrayB, int lengthOfArray, int numberOfThreads);
*/

class Iterator
{
public:
	virtual ~Iterator() = default;
	virtual void next() = 0;
	virtual void *get() = 0;
	virtual std::unique_ptr<Iterator> clone(size_t pos) = 0;
};

LIB_MATCH_EXPORT
class ContiguousMemoryIterator : public Iterator
{
public:
	ContiguousMemoryIterator(void* ptr, int elem_size);
	void next() override;
	void* get() override;
	std::unique_ptr<Iterator> clone(size_t pos) override;
private:
	char *ptr;
	int elem_size;
};

LIB_MATCH_EXPORT
class LibMatchDiagnose
{
public:
	static void getMaxMemoryUsage(size_t *max_memory_size, size_t *max_page_locked_memory_size, size_t *max_gpu_memory_size);
};

template <typename Type>
class BlockMatch
{
public:
	// SearchRegion size 0 for full search
	BlockMatch(std::type_index inputADataType, std::type_index inputBDataType,
		std::type_index outputDataType,
		std::type_index indexDataType,
		SearchType searchType,
		MeasureMethod measureMethod,
		PadMethod padMethodA, PadMethod padMethodB,
		BorderType sequenceABorderType,
		SearchFrom searchFrom,
		bool sort,
		int matrixA_M, int matrixA_N, int matrixB_M, int matrixB_N,
		int searchRegion_M, int searchRegion_N,
		int block_M, int block_N,
		int strideA_M, int strideA_N,
		int strideB_M, int strideB_N,
		int matrixAPadding_M_pre, int matrixAPadding_M_post,
		int matrixAPadding_N_pre, int matrixAPadding_N_post,
		int matrixBPadding_M_pre, int matrixBPadding_M_post,
		int matrixBPadding_N_pre, int matrixBPadding_N_post,
		int numberOfResultsRetain);
	~BlockMatch();
	void initialize();

	void executev2(void *A, void *B,
		Iterator *C,
		void *padded_A = nullptr, void *padded_B = nullptr,
		Iterator *index_x = nullptr, Iterator *index_y = nullptr);
	template <typename InputADataType, typename InputBDataType, typename OutputDataType, typename IndexDataType>
	void execute(InputADataType *A, InputBDataType *B,
		OutputDataType *C,
		InputADataType *padded_A = nullptr, InputBDataType *padded_B = nullptr,
		IndexDataType *index_x = nullptr, IndexDataType *index_y = nullptr)
	{
		if (inputADataType != typeid(InputADataType) || inputBDataType != typeid(InputBDataType)
			|| outputDataType != typeid(OutputDataType)
			|| (index_x && indexDataType != typeid(IndexDataType)))
			throw std::runtime_error("Type check failed");
		execute(static_cast<void*>(A), static_cast<void*>(B),
			static_cast<void*>(C),
			static_cast<void*>(padded_A), static_cast<void*>(padded_B),
			static_cast<void*>(index_x), static_cast<void*>(index_y));
	}
	void execute(void *A, void *B,
		void *C,
		void *padded_A = nullptr, void *padded_B = nullptr,
		void *index_x = nullptr, void *index_y = nullptr);
	void destroy();
	void get_matrixC_dimensions(int *dim0, int *dim1, int *dim2);
	void get_matrixA_padded_dimensions(int *m, int *n);
	void get_matrixB_padded_dimensions(int *m, int *n);
private:
	void *m_instance;
	std::type_index inputADataType;
	std::type_index inputBDataType;
	std::type_index outputDataType;
	std::type_index indexDataType;
};

LIB_MATCH_EXPORT
bool libMatchReset();

LIB_MATCH_EXPORT
void libMatchOnLoad();

LIB_MATCH_EXPORT
void libMatchAtExit();

typedef void LibMatchSinkFunction(const char *);
LIB_MATCH_EXPORT
void libMatchRegisterLoggingSinkFunction(LibMatchSinkFunction sinkFunction);

typedef bool LibMatchInterruptPendingFunction();
LIB_MATCH_EXPORT
void libMatchRegisterInterruptPeddingFunction(LibMatchInterruptPendingFunction);

template <typename T>
void zeroPadding(const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post);
template <typename T>
void circularPadding(const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post);
template <typename T>
void replicatePadding(const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post);
template <typename T>
void symmetricPadding(const T *src_ptr, T *dst_ptr,
	size_t m, size_t n,
	size_t m_pre, size_t m_post, size_t n_pre, size_t n_post);

LIB_MATCH_EXPORT
class InstructionSet
{
	// forward declarations  
	class InstructionSet_Internal;

public:
	// getters  
	static std::string Vendor(void);
	static std::string Brand(void);

	static bool SSE3(void);
	static bool PCLMULQDQ(void);
	static bool MONITOR(void);
	static bool SSSE3(void);
	static bool FMA(void);
	static bool CMPXCHG16B(void);
	static bool SSE41(void);
	static bool SSE42(void);
	static bool MOVBE(void);
	static bool POPCNT(void);
	static bool AES(void);
	static bool XSAVE(void);
	static bool OSXSAVE(void);
	static bool AVX(void);
	static bool F16C(void);
	static bool RDRAND(void);

	static bool MSR(void);
	static bool CX8(void);
	static bool SEP(void);
	static bool CMOV(void);
	static bool CLFSH(void);
	static bool MMX(void);
	static bool FXSR(void);
	static bool SSE(void);
	static bool SSE2(void);

	static bool FSGSBASE(void);
	static bool BMI1(void);
	static bool HLE(void);
	static bool AVX2(void);
	static bool BMI2(void);
	static bool ERMS(void);
	static bool INVPCID(void);
	static bool RTM(void);
	static bool AVX512F(void);
	static bool RDSEED(void);
	static bool ADX(void);
	static bool AVX512PF(void);
	static bool AVX512ER(void);
	static bool AVX512CD(void);
	static bool SHA(void);

	static bool PREFETCHWT1(void);

	static bool LAHF(void);
	static bool LZCNT(void);
	static bool ABM(void);
	static bool SSE4a(void);
	static bool XOP(void);
	static bool TBM(void);

	static bool SYSCALL(void);
	static bool MMXEXT(void);
	static bool RDTSCP(void);
	static bool _3DNOWEXT(void);
	static bool _3DNOW(void);

private:
	static const InstructionSet_Internal CPU_Rep;

	class InstructionSet_Internal
	{
	public:
		InstructionSet_Internal();

		int nIds_;
		int nExIds_;
		std::string vendor_;
		std::string brand_;
		bool isIntel_;
		bool isAMD_;
		std::bitset<32> f_1_ECX_;
		std::bitset<32> f_1_EDX_;
		std::bitset<32> f_7_EBX_;
		std::bitset<32> f_7_ECX_;
		std::bitset<32> f_81_ECX_;
		std::bitset<32> f_81_EDX_;
		std::vector<std::array<int, 4>> data_;
		std::vector<std::array<int, 4>> extdata_;
	};
};

enum class memory_type {
	system,
	page_locked,
	gpu
};

LIB_MATCH_EXPORT
std::string to_string(memory_type type);

LIB_MATCH_EXPORT
class memory_alloc_exception : public std::runtime_error
{
public:
	memory_alloc_exception(const std::string& _Message,
		memory_type type,
		size_t max_memory_size, size_t max_page_locked_memory_size, size_t max_gpu_memory_size,
		size_t current_memory_size, size_t current_page_locked_memory_size, size_t current_gpu_memory_size);

	memory_alloc_exception(const char* _Message,
		memory_type type,
		size_t max_memory_size, size_t max_page_locked_memory_size, size_t max_gpu_memory_size,
		size_t current_memory_size, size_t current_page_locked_memory_size, size_t current_gpu_memory_size);

	memory_type get_memory_allocation_type() const;

	size_t get_max_memory_size() const;
	size_t get_max_page_locked_memory_size() const;
	size_t get_max_gpu_memory_size() const;
	size_t get_current_memory_size() const;
	size_t get_current_page_locked_memory_size() const;
	size_t get_current_gpu_memory_size() const;

private:
	memory_type type;
	size_t max_memory_size;
	size_t max_page_locked_memory_size;
	size_t max_gpu_memory_size;
	size_t current_memory_size;
	size_t current_page_locked_memory_size;
	size_t current_gpu_memory_size;
};

int getTypeSize(std::type_index type);