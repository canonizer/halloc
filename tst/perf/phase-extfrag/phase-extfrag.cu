/** @file phase-extfrag.cu probabalitized external fragmentation test */

#include <common.h>

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** global counters for number of allocations, frees and total size allocated
		*/
__device__ uint64 nmallocs_g = 0;

/** the kernel of the probability throughput test */
template <class T>
__global__ void prob_throughput_k
(void **ptrs, uint *ctrs, uint itry) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	uint n = opts_g.nthreads;
	//uint nallocs = opts_g.nallocs;
	if(opts_g.is_thread_inactive(i))
		return;
	uint ctr = ctrs[i];
	//uint nmallocs = 0;

	// iterate 
	for(uint iter = 0; iter < opts_g.niters; iter++) {
		// perform the action
		switch(opts_g.next_action(ctr > 0, itry, iter)) {
			//switch(ctr > 0 ? ActionFree : ActionAlloc) {
		case ActionAlloc:
			for(uint ialloc = 0; ialloc < opts_g.nallocs; ialloc++)
				ptrs[ialloc * n + i] = T::malloc(opts_g.next_alloc_sz());
			ctr = opts_g.nallocs;
			//nmallocs += nallocs;
			break;
		case ActionFree:
			for(uint ialloc = 0; ialloc < opts_g.nallocs; ialloc++)
				T::free(ptrs[ialloc * n + i]);
			ctr = 0;
			break;
		case ActionNone:
			//printf("no action taken\n");
			break;
		}
	}  // for(each iteration)
	ctrs[i] = ctr;
}  // prob_throughput_k

/** measures malloc throughput */
template<class T> class PhaseExtFragTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		// allocate memory
		if(warmup) {
			opts.nthreads = min(4 * opts.bs, opts.nthreads);
			opts.ntries = 1;
			opts.niters = 1;
		}
		if(!warmup)
			printf("two-phase throuhgput test\n");
		cuset(opts_g, CommonOpts, opts);
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
			// run the kernel
			prob_throughput_k<T> <<<grid, bs>>>(d_ptrs, d_ctrs, itry);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));			
			// check that pointers are correct
			if(!check_nz(d_ptrs, d_ctrs, nptrs, opts)) {
				fprintf(stderr, "cannot allocate enough memory\n");
				exit(-1);
			}
			if(!warmup)
				printf("external fragmentation %d %.2lf %.2lf\n", itry, T::extfrag(false),
							 T::extfrag(true));
		}  // for(itry)

		// free the rest
		{
			free_rest_k<T> <<<grid, bs>>> (/* opts, */ d_ptrs, d_ctrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
		}

		// free memory
		cucheck(cudaFree(d_ptrs));
		cucheck(cudaFree(d_ctrs));
	}  // operator()
 
};  // PhaseExtFragTest

int main(int argc, char **argv) {
	CommonOpts opts(true);
	run_test<PhaseExtFragTest>(argc, argv, opts);
	return 0;
}  // main
