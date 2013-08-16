#ifndef HALLOC_COMMON_H_
#define HALLOC_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
			halloc_fraction(0.75), busy_fraction(0.82), roomy_fraction(0.25),
			sparse_fraction(0.05), sb_sz_sh(22), nthreads(1024 * 1024), ntries(8),
			alloc_sz(16), nallocs(4), alloc_fraction(0.4), bs(256), period_mask(0) { }
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
	/** slab occupancy below which it is declared sparse, -S */
	double sparse_fraction;
	/** shift of slab size, -b */
	int sb_sz_sh;
	
	// test parameters
	/** number of threads in the test, -n */
	int nthreads;
	/** thread block size, currently only default is available without any options
	*/
	int bs;
	/** number of tries in the test, -t */
	int ntries;
	/** allocation size in bytes when fixed, -s */
	uint alloc_sz;
	/** number of allocations per thread, -l */
	int nallocs;
	/** fraction of memory to allocate in test, -f */
	double alloc_fraction;
	/** period mask, indicates one of how many threads actually does allocation;
	-p specifies period shift
	*/
	int period_mask;
	/** gets the total number of allocations, as usually defined for tests;
			individual tests may use their own definition */
	double total_nallocs(void);
};

/** checks that all the pointers are non-zero 
		@param d_ptrs device pointers
		@param nptrs the number of pointers
		@param period the step with which to check values
 */
bool check_nz(void **d_ptrs, int nptrs, int period);

#include "halloc-wrapper.h"
#include "cuda-malloc-wrapper.h"

/** a kernel (and function) for warming up the allocator; a number of memory
		allocations with a small number of threads are made; the allocations are
		then freed, with no measurements performed */
template<class T>
__global__ void warm_up_k(uint alloc_sz) {
	void *p = T::malloc(alloc_sz);
	T::free(p);
}  // warm_up_k

template<class T>
void warm_up(void) {
	int nthreads = 4, bs = 256;
	for(uint alloc_sz = 16; alloc_sz < 64; alloc_sz += 8) {
		warm_up_k<T> <<<divup(nthreads, bs), bs>>>(alloc_sz);
		cucheck(cudaGetLastError());
	}
	cucheck(cudaStreamSynchronize(0));
}

/** does a test with specific allocator and test functor; it is called after
		command line parsing */
template <class T, template<class Ta> class Test>
void run_test(CommonOpts &opts) {
	T::init(opts);
	//warm_up<T>();
	
	Test<T> test;
	// warmup
	test(opts, true);
	// real run
	test(opts, false);

	T::shutdown();
}  // run_test

/** does a test with specific test functor; basically
		this is a main function for all the tests */
template <template<class Ta> class Test >
void run_test(int argc, char ** argv, CommonOpts &opts) {
	// parse command line
	opts.parse_cmdline(argc, argv);

	// instantiate based on allocator type
	switch(opts.allocator) {
	case AllocatorCuda:
		run_test <class CudaMalloc, Test> (opts);
		break;
	case AllocatorHalloc:
		//printf("testing halloc allocator\n");
		run_test <class Halloc, Test> (opts);
		break;
	default:
		fprintf(stderr, "allocator invalid or not supported\n");
		exit(-1);
	}
}  // run_test

#endif
