/** @file prob-throughput.cu probabalitized throughput test */

#include <common.h>

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** the kernel of the probability throughput test */
template <class T>
__global__ void prob_corr_k
(void **ptrs, uint *ctrs, uint itry) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	uint n = opts_g.nthreads, nallocs = opts_g.nallocs;
	if(opts_g.is_thread_inactive(i))
		return;
	uint ctr = ctrs[i];

	// iterate 
	for(uint iter = 0; iter < opts_g.niters; iter++) {
		// perform the action
		switch(opts_g.next_action(ctr > 0, itry, iter)) {
		case ActionAlloc:
			for(uint ialloc = 0; ialloc < nallocs; ialloc++) {
				uint sz = opts_g.next_alloc_sz();
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

		cuset(opts_g, CommonOpts, opts);

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			printf("iteration %d\n", itry);
			// run the kernel
			//printf("kernel configuration: %d, %d\n", grid, bs);
			prob_corr_k<T> <<<grid, bs>>>(d_ptrs, d_ctrs, itry);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			// check that pointers are correct
			if(!check_alloc(d_ptrs, d_ctrs, nptrs, opts)) {
			 	fprintf(stderr, "cannot allocate enough memory\n");
			 	exit(-1);
			}
		}  // for(itry)

		// free the rest
		printf("freeing the rest\n");
		free_rest_k<T> <<<grid, bs>>> (d_ptrs, d_ctrs);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));

		// free memory
		cucheck(cudaFree(d_ptrs));
		cucheck(cudaFree(d_ctrs));
	}  // operator()
 
};  // ProbThroughputTest

int main(int argc, char **argv) {
	CommonOpts opts(true);
	run_test<ProbCorrTest>(argc, argv, opts, false);
	return 0;
}  // main
