/** @file latency.cu latency test for various memory allocators */

#include <common.h>

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** measures malloc latencies; note that latencies are averaged per-thread,
		per-allocation latencies are not preserved; latencies here are measured in cycles */
template<class T>
__global__ void throughput_malloc_k(CommonOpts opts, void **ptrs) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= n || i & opts.period_mask)
		return;
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) 
		ptrs[i + n * ialloc] = T::malloc(opts.alloc_sz);
}  // latency_malloc_k

template<class T> 
__global__ void throughput_free_k(CommonOpts opts, void **ptrs) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= n || i & opts.period_mask)
		return;
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) 
		T::free(ptrs[i + n * ialloc]);
}  // latency_free_k

template<class T> class CheckPtrTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		// allocate memory
		if(warmup) {
			opts.nthreads = min(4 * opts.bs, opts.nthreads);
			opts.ntries = 1;
		}
		int n = opts.nthreads, bs = opts.bs, grid = divup(n, bs);
		int nptrs = n * opts.nallocs;
		size_t ptrs_sz = nptrs * sizeof(void *);
		void **d_ptrs;
		cucheck(cudaMalloc((void **)&d_ptrs, ptrs_sz));
		cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			// allocate
			throughput_malloc_k<T> <<<grid, bs>>>(opts, d_ptrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			// check that pointers are correct
			if(!check_alloc(d_ptrs, opts.alloc_sz, nptrs, opts.period_mask + 1)) {
				exit(-1);
			}
			// free
			throughput_free_k<T> <<<grid, bs>>>(opts, d_ptrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
		}  // for(itry)

		// free memory
		cucheck(cudaFree(d_ptrs));		
	}  // operator()
 
};  // CheckPtrTest

int main(int argc, char **argv) {
	CommonOpts opts;
	run_test<CheckPtrTest> (argc, argv, opts);
	return 0;
}  // main
