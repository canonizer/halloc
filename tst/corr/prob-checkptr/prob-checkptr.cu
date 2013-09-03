/** @file prob-throughput.cu probabalitized throughput test */

#include <common.h>

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** global counters for number of allocations, frees and total size allocated
		*/

/** the kernel of the probability throughput test */
template <class T>
__global__ void prob_corr_k(CommonOpts opts, void **ptrs, uint *ctrs) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	uint n = opts.nthreads, nallocs = opts.nallocs;
	if(opts.is_thread_inactive(i))
		return;
	uint ctr = ctrs[i];

	// iterate 
	for(uint iter = 0; iter < opts.niters; iter++) {
		// get the action
		float rnd = drandomf();
		ActionType action = ActionNone;
		if(ctr == 0 && rnd <= opts.palloc) {
			action = ActionAlloc;
		} else if(ctr == opts.nallocs && rnd <= opts.pfree) {
			action = ActionFree;
		}
		// perform the action
		switch(action) {
		case ActionAlloc:
			for(uint ialloc = 0; ialloc < nallocs; ialloc++) {
				uint sz = opts.next_alloc_sz();
				void *ptr = T::malloc(sz);
				ptrs[ialloc * n + i] = ptr;
				if(ptr)
					*(uint *)ptr = sz;
				//printf("tid = %d, sz = %d\n", i, sz);
			}
			ctr = nallocs;
			break;
		case ActionFree:
			for(uint ialloc = 0; ialloc < nallocs; ialloc++)
				T::free(ptrs[ialloc * n + i]);
			ctr = 0;
			break;
		}
	}  // for(each iteration)
	ctrs[i] = ctr;
}  // prob_throughput_k

/** free the rest after the throughput test; this also counts against the total
		time */
template <class T> __global__ void free_rest_k
(CommonOpts opts, void **ptrs, uint *ctrs) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	uint ctr = ctrs[i], n = opts.nthreads;
	for(uint ialloc = 0; ialloc < ctr; ialloc++) {
		T::free(ptrs[n * ialloc + i]);
	}
	ctrs[i] = 0;
}  // free_rest_k

/** measures malloc throughput */
template<class T> class ProbCorrTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		// allocate memory
		int n = opts.nthreads, bs = opts.bs, grid = divup(n, bs);
		int nptrs = n * opts.nallocs;
		size_t ptrs_sz = nptrs * sizeof(void *);
		uint ctrs_sz = n * sizeof(uint);
		void **d_ptrs;
		cucheck(cudaMalloc((void **)&d_ptrs, ptrs_sz));
		cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));
		uint *d_ctrs;
		cucheck(cudaMalloc((void **)&d_ctrs, ctrs_sz));
		cucheck(cudaMemset(d_ctrs, 0, ctrs_sz));

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			//printf("iteration %d\n", itry);
			// run the kernel
			//printf("kernel configuration: %d, %d\n", grid, bs);
			prob_corr_k<T> <<<grid, bs>>>(opts, d_ptrs, d_ctrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			// check that pointers are correct
			if(!check_alloc(d_ptrs, d_ctrs, nptrs, opts)) {
			 	fprintf(stderr, "cannot allocate enough memory\n");
			 	exit(-1);
			}
		}  // for(itry)

		// free the rest
		//printf("freeing the rest\n");
		free_rest_k<T> <<<grid, bs>>> (opts, d_ptrs, d_ctrs);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));

		// free memory
		cucheck(cudaFree(d_ptrs));
		cucheck(cudaFree(d_ctrs));
	}  // operator()
 
};  // ProbThroughputTest

int main(int argc, char **argv) {
	CommonOpts opts;
	run_test<ProbCorrTest>(argc, argv, opts, false);
	return 0;
}  // main
