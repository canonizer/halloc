/** @file freeslabs.cu tests whether all slabs are returned as free */

#include <common.h>

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

template<class T> class FreeSlabsTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		opts.niters = 1;
		// allocate memory
		if(warmup) {
			opts.nthreads = min(4 * opts.bs, opts.nthreads);
			opts.ntries = 1;
		}
		// override number of allocations, period and group options
		opts.nallocs = 1;
		opts.period_mask = 0;
		opts.group_sh = 0;
		int max_n = opts.nthreads, nptrs = max_n * opts.nallocs;
		// note that here, nthreads is treated as the maximum thread number
		size_t ptrs_sz = nptrs * sizeof(void *);
		void **d_ptrs;
		cucheck(cudaMalloc((void **)&d_ptrs, ptrs_sz));
		cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));

		// allocation fraction; increase to larger values when it's possible 
		// to free cached or head slabs
		double fraction = 0.4;
		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			// step over sizes:
			// 16..64: step 8
			// 64..256: step 16
			// 256..1k: step 128
			uint step = 8;
			for(uint alloc_sz = 16; alloc_sz <= 1024; alloc_sz += step) {
				printf("allocation size %d\n", alloc_sz);
				int nthreads = (int)floor(fraction * opts.memory / alloc_sz);
				nthreads = min(max_n, nthreads);
				opts.nthreads = nthreads;
				opts.alloc_sz = opts.max_alloc_sz = alloc_sz;
				opts.recompute_fields();
				int bs = opts.bs, grid = divup(opts.nthreads, bs);
				// allocate
				malloc_k<T> <<<grid, bs>>>(opts, d_ptrs);
				cucheck(cudaGetLastError());
				cucheck(cudaStreamSynchronize(0));
				// check that pointers are correct
				if(!check_alloc(d_ptrs, 0, opts.nthreads, opts)) {
					exit(-1);
				}
				// free
				free_k<T> <<<grid, bs>>>(opts, d_ptrs);
				cucheck(cudaGetLastError());
				cucheck(cudaStreamSynchronize(0));
				// set up step
				if(alloc_sz >= 256)
					step = 128;
				else if(alloc_sz >= 64)
					step = 16;
				else
					step = 8;
			}  // for(alloc_sz)
		}  // for(itry)

		// free memory
		cucheck(cudaFree(d_ptrs));		
	}  // operator()
 
};  // FreeSlabsTest

int main(int argc, char **argv) {
	CommonOpts opts(true);
	opts.ntries = 4;
	run_test<FreeSlabsTest> (argc, argv, opts, false);
	return 0;
}  // main
