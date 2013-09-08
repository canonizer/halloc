/** @file latency.cu latency test for various memory allocators */

#include <common.h>

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

template<class T> class CheckPtrTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		opts.niters = 1;
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
			malloc_corr_k<T> <<<grid, bs>>>(opts, d_ptrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			// check that pointers are correct
			if(!check_alloc(d_ptrs, 0, nptrs, opts)) {
				exit(-1);
			}
			// free
			free_k<T> <<<grid, bs>>>(opts, d_ptrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
		}  // for(itry)

		// free memory
		cucheck(cudaFree(d_ptrs));		
	}  // operator()
 
};  // CheckPtrTest

int main(int argc, char **argv) {
	CommonOpts opts(true);
	run_test<CheckPtrTest> (argc, argv, opts, false);
	return 0;
}  // main
