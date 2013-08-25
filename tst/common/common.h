#ifndef HALLOC_COMMON_H_
#define HALLOC_COMMON_H_

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

/** division with rounding upwards, useful for kernel calls */
inline int divup(int a, int b) { return a / b + (a % b ? 1 : 0); }

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

/** common options for tests and allocator intiialization; note that some tests
		are free to provide their own default settings */
struct CommonOpts {
	/** default initialization for common options */
	CommonOpts() 
		: allocator(AllocatorHalloc), memory(512 * 1024 * 1024), 
			halloc_fraction(0.75), busy_fraction(0.87), roomy_fraction(0.6),
			sparse_fraction(0.05), sb_sz_sh(22), device(0), nthreads(1024 * 1024), 
			ntries(8), alloc_sz(16), max_alloc_sz(16), nallocs(4),
			alloc_fraction(0.4), bs(256), period_mask(0), group_sh(0) { }
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
	/** number of allocations per thread, -l */
	int nallocs;
	/** fraction of memory to allocate in test, -f */
	double alloc_fraction;
	/** period mask, indicates one of how many threads actually does allocation;
	-p specifies period shift
	*/
	int period_mask;
	/** group size for period; the "period" parameter is applied to groups, not
	individual threads; -g */
	int group_sh;
	/** gets the total number of allocations, as usually defined for tests; for
	randomized tests, expectation is returned; individual tests may use their own
	definition */
	double total_nallocs(void);
	/** gets the total size of all the allocations; for randomized tests,
	expectation is returned
	*/
	double total_sz(void);
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
};

/** checks that all the pointers are non-zero 
		@param d_ptrs device pointers
		@param nptrs the number of pointers
 */
bool check_nz(void **d_ptrs, uint nptrs, const CommonOpts &opts);

/** checks that all allocations are made properly, i.e. that no pointer is zero,
		and there's at least alloc_sz memory after each pointer (alloc_sz is the
		same for all allocations). Parameters are mostly the same as with check_nz()
  */
bool check_alloc(void **d_ptrs, uint nptrs, const CommonOpts &opts);

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

	// instantiate based on allocator type
	switch(opts.allocator) {
	case AllocatorCuda:
		run_test <class CudaMalloc, Test> (opts, with_warmup);
		break;
	case AllocatorHalloc:
		//printf("testing halloc allocator\n");
		run_test <class Halloc, Test> (opts, with_warmup);
		break;
	case AllocatorScatterAlloc:
		run_test <class ScatterAlloc, Test> (opts, with_warmup);
		break;
	default:
		fprintf(stderr, "allocator invalid or not supported\n");
		exit(-1);
	}
}  // run_test

/** helper malloc kernel used by many tests throughout */
template<class T>
__global__ void malloc_k
(CommonOpts opts, void **ptrs) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) 
		ptrs[i + n * ialloc] = T::malloc(opts.alloc_sz);
}  // throughput_malloc_k

/** helper free kernel used by many tests throughout */
template<class T>
__global__ void free_k
(CommonOpts opts, void **ptrs) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) 
		T::free(ptrs[i + n * ialloc]);
}  // throughput_free_k

#endif
