#ifndef HALLOC_COMMON_H_
#define HALLOC_COMMON_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** number of bytes in one GiB */
#define NBYTES_IN_GIB (1024.0 * 1024.0 * 1024.0)

/** a macro for checking CUDA calls */
#define cucheck(call)																										\
	{																																			\
	cudaError_t cucheck_err = (call);																			\
	if(cucheck_err != cudaSuccess) {																			\
		const char* err_str = cudaGetErrorString(cucheck_err);							\
		fprintf(stderr, "%s (%d): %s in %s\n", __FILE__, __LINE__, err_str, #call);	\
		exit(-1);																														\
	}																																			\
	}

/** sets CUDA device variable */
#define cuset(symbol, T, val)																		\
{																																\
	void *cuset_addr;																							\
	cucheck(cudaGetSymbolAddress(&cuset_addr, symbol));						\
	T cuset_val = (val);																					\
	cucheck(cudaMemcpy(cuset_addr, &cuset_val, sizeof(cuset_val), \
										 cudaMemcpyHostToDevice));									\
}  // cuset

/** gets the value of the CUDA device variable */
#define cuget(pval, symbol)																\
{																													\
	void *cuget_addr;																				\
	cucheck(cudaGetSymbolAddress(&cuget_addr, symbol));			\
	cucheck(cudaMemcpy((pval), cuget_addr, sizeof(*(pval)), \
										 cudaMemcpyDeviceToHost));						\
}

/** division with rounding upwards, useful for kernel calls */
inline int divup(int a, int b) { return a / b + (a % b ? 1 : 0); }

/** short-name typedef for a long long unsigned type */
typedef unsigned long long uint64;

/** @file common.h common functions and definitions for testing infrastructure
		of halloc and similar GPU memory allocations. Note that this is provided for
		testing, performance measurement and comparison only, and is not intended
		for use in end-user applications. For end-user applications, halloc or
		another allocator is better to be used directly. */

/** supported allocators */
typedef enum {
	AllocatorNone = 0, AllocatorCuda, AllocatorHalloc, AllocatorScatterAlloc, 
	AllocatorXMalloc, AllocatorTopNone
} AllocatorType;

/** supported allocation size distributions */
typedef enum {
	DistrNone = 0, DistrUniform, DistrExpUniform, DistrExpEqual, DistrTopNone
} DistrType;

/** allocation action */
typedef enum {
	ActionNone = 0, ActionAlloc, ActionFree
} ActionType;

#ifdef COMMONTEST_COMPILING
#define COMMONTEST_EXTERN
#else
#define COMMONTEST_EXTERN extern
#endif

/** external variable holding random values, one per thread */
COMMONTEST_EXTERN uint * __constant__ random_states_g;

/** get the random value on the device */
static inline  __device__ uint drandom(void) {
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint seed = random_states_g[tid];
	// TODO: check if other advancements algorithms are faster
	/* seed ^= (seed << 13);
	seed ^= (seed >> 17);
	seed ^= (seed << 5); */
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	random_states_g[tid] = seed;
	return seed;
}  // drandom

/** get the random value within the specified interval (both ends inclusive) on
		the device */
static inline __device__ uint drandom(uint a, uint b) {
	return a + (drandom() & 0x00ffffffu) % (uint)(b - a + 1);
}  // drandom

/** get the floating-point random value between 0 and 1 */
static inline __device__ float drandomf(void) {
	float f = 1.0f / (1024.0f * 1024.0f);
	uint m = 1024 * 1024;
	return f * drandom(0, m - 1);
}  // drandomf

/** get the random boolean value with the specified probability
		@param probab the probability to return true
 */
static inline __device__ bool drandomb(float probab) {
	if(0.0f < probab && probab < 1.0f)
		return drandomf() <= probab;
	else 
		return probab >= 1.0f;
}  // drandomb

/** common options for tests and allocator intiialization; note that some tests
		are free to provide their own default settings */
struct CommonOpts {
	/** default initialization for common options */
	CommonOpts(bool dummy) 
		: allocator(AllocatorHalloc), memory(512 * 1024 * 1024), 
			halloc_fraction(0.75), busy_fraction(0.835), roomy_fraction(0.6),
			sparse_fraction(0.0125), sb_sz_sh(22), device(0), nthreads(1024 * 1024),
			ntries(8), alloc_sz(16), max_alloc_sz(16), nallocs(4), niters(1),
			bs(128), period_mask(0), group_sh(0),	distr_type(DistrUniform), 
			alloc_fraction(1), free_fraction(0), exec_fraction(1) {
		recompute_fields();
	}

	__host__ __device__ CommonOpts() {}
	/** parses the options from command line, with the defaults specified; memory
		is also capped to fraction of device-available at this step 
		@param [in, out] this the default options on the input, and the options
		provided by the command line on the output
	*/
	void parse_cmdline(int argc, char **argv);
	/** the allocator type, as parsed from the command line, -a */
	AllocatorType allocator;
	// allocator arguments
	/** maximum allocatable memory; silently capped by a fraction (0.75) of
			available device memory, -m */
	size_t memory;
	/** fraction of memory allocated for halloc allocator, halloc only, -C */
	double halloc_fraction;
	/** slab occupancy above which it is declared busy, -B */
	double busy_fraction;
	/** slab occupancy below which it is declared roomy, -R */
	double roomy_fraction;
	/** slab occupancy below which it is declared sparse; currently, no option, as
		we don't see where it's useful */
	double sparse_fraction;
	/** shift of slab size, -b */
	int sb_sz_sh;
	
	// test parameters
	/** the device on which everything runs, -D */
	int device;
	/** number of threads in the test, -n */
	int nthreads;
	/** thread block size, -T	*/
	int bs;
	/** number of tries in the test, -t */
	int ntries;
	/** allocation size in bytes when fixed, -s */
	uint alloc_sz;
	/** maximum alloc size in bytes, -S */
	uint max_alloc_sz;
	/** ceil(log2(max_alloc_sz/alloc_sz) */
	uint max_alloc_sh;
	/** number of allocations per thread, -l */
	int nallocs;
	/** number of inside-kernel iterations, applicable only to priv-* samples,
	forced to one in other cases, -i */
	int niters;
	/** period mask, indicates one of how many threads actually does allocation;
	-q specifies period shift
	*/
	int period_mask;
	/** group size for period; the "period" parameter is applied to groups, not
	individual threads; -g */
	int group_sh;
	/** gets the allocation size distribution type; -d */
	DistrType distr_type;
	/** probabilities; first dimension is the phase (alloc = 0, free = 1), second
	dimension is the action to be taken (alloc = 0, free = 1); these cannot be specified
	from command line directly, and computed instead from steady state*/
	float probabs[2][2];
	/** the steady state fraction threads having something allocated after the
	allocation phase (f' in equation terms); -f
	*/
	float alloc_fraction;
	/** the steady state fraction of threads having something allocated after the
	free phase (f'' in equation terms); -F */
	float free_fraction;
	/** the fraction of threads which need to do (execute) something between
	steady states; -e */
	float exec_fraction;
	/** gets the total number of allocations, as usually defined for tests; for
	randomized tests, expectation is returned; individual tests may use their own
	definition */
	double total_nallocs(void);
	/** gets the total size of all the allocations; for randomized tests,
	expectation is returned
	*/
	double total_sz(void);
	/** gets the single allocation expectation size */
	double expected_sz(void);

	/** gets the next action */
	__device__ ActionType next_action
	(bool allocated, uint itry, uint iter) const {
		uint phase = (itry * niters + iter) % 2;
		uint state = allocated ? 1 : 0;
		if(drandomb(probabs[phase][state]))
			return allocated ? ActionFree : ActionAlloc;
		else
			return ActionNone;
	}  // next_action

	/** gets the next allocation size, which can be random */
	__device__ uint next_alloc_sz(void) const {
		// single-size case
		if(!is_random())
			return alloc_sz;
		switch(distr_type) {
		case DistrUniform:
			{
				uint sz = drandom(alloc_sz, max_alloc_sz);
				//sz = min(sz, max_alloc_sz);
				//printf("sz = %d, alloc_sz = %d, max_alloc_sz = %d\n", sz, alloc_sz, 
				//			 max_alloc_sz);
				return sz;
			}
		case DistrExpUniform:
			{
				// get random shift
				uint sh = drandom(0, max_alloc_sh);
				// get a value within the exponential group
				uint sz = drandom(alloc_sz << sh, (alloc_sz << (sh + 1)) - 1);
				sz = min(sz, max_alloc_sz);
				return sz;
			}
		case DistrExpEqual:
			{
				// get shift, distributed in geometric progression (shift *2 =>
				// probability / 2)
				uint sh = __ffs(drandom(1, 1 << (max_alloc_sh + 1))) - 1;
				// get a value within the exponential group
				uint sz = drandom(alloc_sz << sh, (alloc_sz << (sh + 1)) - 1);
				sz = min(sz, max_alloc_sz);
				return sz;
			}
		default:
			// this should definitely not happen
			assert(0);
			return 0;
		}
	}  // next_alloc_sz
	/** checks whether the thread is inactive */
	__host__ __device__ bool is_thread_inactive(uint tid) const {
		return tid >= nthreads || (tid >> group_sh) & period_mask;
	}
	/** gets the period */
	__host__ __device__ uint period(void) const { return period_mask + 1; }
	/** gets the group size */
	__host__ __device__ uint group(void) const { return 1 << group_sh; }
	/** gets the (contiguous) number of pointers for the given number of threads */
	__host__ __device__ uint nptrs_cont(uint nts) const {
		return nts / (group() * period()) * group() + 
			min(nts % (group() * period()), group());
	}
	/** checks whether randomization is employed */
	__host__ __device__ uint is_random(void) const {
		return alloc_sz != max_alloc_sz;
	}
	/** recompute the fields which need be recomputed */
	void recompute_fields(void);
};

#ifndef COMMONTEST_COMPILING
__constant__ CommonOpts opts_g;
#endif

/** initialize device generation of random numbers */
void drandom_init(const CommonOpts &opts);

/** shutdown device generation of random numbers */
void drandom_shutdown(const CommonOpts &opts);

/** checks that all the pointers are non-zero 
		@param d_ptrs device pointers
		@param nptrs the number of pointers
 */
bool check_nz(void **d_ptrs, uint *d_ctrs, uint nptrs, const CommonOpts &opts);

/** checks that all allocations are made properly, i.e. that no pointer is zero,
		and there's at least alloc_sz memory after each pointer (alloc_sz is the
		same for all allocations). Parameters are mostly the same as with check_nz()
  */
bool check_alloc(void **d_ptrs, uint *d_ctrs, uint nptrs,
								 const CommonOpts &opts);

#include "halloc-wrapper.h"
#include "cuda-malloc-wrapper.h"
#include "scatter-alloc-wrapper.h"

/** does a test with specific allocator and test functor; it is called after
		command line parsing */
template <class T, template<class Ta> class Test>
void run_test(CommonOpts &opts, bool with_warmup) {
	T::init(opts);
	//warm_up<T>();
	
	Test<T> test;
	// warmup, if necessary
	if(with_warmup)
		test(opts, true);
	// real run
	test(opts, false);

	T::shutdown();
}  // run_test

/** does a test with specific test functor; basically
		this is a main function for all the tests */
template <template<class Ta> class Test >
void run_test(int argc, char ** argv, CommonOpts &opts, bool with_warmup = true) {
	// parse command line
	opts.parse_cmdline(argc, argv);
	cucheck(cudaSetDevice(opts.device));

	// initialize random numbers
	drandom_init(opts);

	// instantiate based on allocator type
	switch(opts.allocator) {
	case AllocatorCuda:
		run_test <class CudaMalloc, Test> (opts, with_warmup);
		break;
	case AllocatorHalloc:
		//printf("testing halloc allocator\n");
		run_test <class Halloc, Test> (opts, with_warmup);
		break;
#ifdef WITH_SCATTER
	case AllocatorScatterAlloc:
		run_test <class ScatterAlloc, Test> (opts, with_warmup);
		break;
#endif
	default:
		fprintf(stderr, "allocator invalid or not supported\n");
		exit(-1);
	}
}  // run_test

#ifndef COMMONTEST_COMPILING

/** helper malloc kernel used by many tests throughout */
template<class T>
__global__ void malloc_k
(CommonOpts opts, void **ptrs) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) {
		uint sz = opts.next_alloc_sz();
		void *ptr = T::malloc(sz);
		ptrs[i + n * ialloc] = ptr;
	}
}  // malloc_k

/** helper non-randomized malloc kernel */
template<class T>
__global__ void malloc_corr_k
(CommonOpts opts, void **ptrs) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) {
		uint sz = opts.next_alloc_sz();
		void *ptr = T::malloc(sz);
		ptrs[i + n * ialloc] = ptr;
		if(ptr)
			*(uint *)ptr = sz;
	}
}  // malloc_corr_k

/** helper free kernel used by many tests throughout */
template<class T>
__global__ void free_k
(CommonOpts opts, void **ptrs) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) 
		T::free(ptrs[i + n * ialloc]);
}  // free_k

/** free the rest after the throughput test; this also counts against the total
		time */
template <class T> __global__ void free_rest_k(void **ptrs, uint *ctrs) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts_g.is_thread_inactive(i))
		return;
	uint ctr = ctrs[i], n = opts_g.nthreads;
	for(uint ialloc = 0; ialloc < ctr; ialloc++) {
		T::free(ptrs[n * ialloc + i]);
	}
	ctrs[i] = 0;
}  // free_rest_k

#endif

#endif
