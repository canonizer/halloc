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
__device__ uint64 nmallocs_g = 0;

/** the kernel of the probability throughput test */
template <class T>
__global__ void prob_throughput_k(CommonOpts opts, void **ptrs, uint *ctrs) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	uint n = opts.nthreads, nallocs = opts.nallocs;
	if(opts.is_thread_inactive(i))
		return;
	uint ctr = ctrs[i], nmallocs = 0;

	// iterate 
	for(uint iter = 0; iter < opts.niters; iter++) {
		// get the action
		float rnd = drandomf();
		//float rnd = iter ? 0.0f : (i / 32 % 2 ? 0.0f : 1.5f);
		ActionType action = ActionNone;
		/*
			if(rnd < opts.palloc) {
			if(ctr < opts.nallocs)
			action = ActionAlloc;
			} else if(rnd < opts.palloc + opts.pfree) {
			if(ctr > 0)
			action = ActionFree;
			}
		*/
		if(ctr == 0 && rnd <= opts.palloc) {
			action = ActionAlloc;
		} else if(ctr == opts.nallocs && rnd <= opts.pfree) {
			action = ActionFree;
		}
		// perform the action
		/*
			switch(action) {
			case ActionAlloc:
			ptrs[ctr++ * n + i] = T::malloc(opts.next_alloc_sz());
			nmallocs++;
			break;
			case ActionFree:
			T::free(ptrs[--ctr * n + i]);
			break;
			}  // switch(action)
		*/
		switch(action) {
		case ActionAlloc:
			for(uint ialloc = 0; ialloc < nallocs; ialloc++)
				ptrs[ialloc * n + i] = T::malloc(opts.next_alloc_sz());
			ctr = nallocs;
			nmallocs += nallocs;
			break;
		case ActionFree:
			for(uint ialloc = 0; ialloc < nallocs; ialloc++)
				T::free(ptrs[ialloc * n + i]);
			ctr = 0;
			break;
		}
	}  // for(each iteration)

	ctrs[i] = ctr;
	// increment counters
	//atomicAdd(&nmallocs_g, nmallocs);
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
template<class T> class ProbThroughputTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		// allocate memory
		if(warmup) {
			opts.nthreads = min(4 * opts.bs, opts.nthreads);
			opts.ntries = 1;
			opts.niters = 1;
		}
		if(!warmup)
			printf("probability throuhgput test\n");
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
		
		double t_pair = 0;
		cuset(nmallocs_g, uint64, 0);

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			// run the kernel
			double t_start = omp_get_wtime();
			prob_throughput_k<T> <<<grid, bs>>>(opts, d_ptrs, d_ctrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			double t_end = omp_get_wtime();
			t_pair += t_end - t_start;
			// check that pointers are correct
			if(!check_nz(d_ptrs, d_ctrs, nptrs, opts)) {
				fprintf(stderr, "cannot allocate enough memory\n");
				exit(-1);
			}		
		}  // for(itry)

		// free the rest
		{
			double t_start = omp_get_wtime();
			free_rest_k<T> <<<grid, bs>>> (opts, d_ptrs, d_ctrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			double t_end = omp_get_wtime();
			t_pair += t_end - t_start;
		}

		// output throughput infos
		if(!warmup) {
			//uint64 nallocs;
			//cuget(&nallocs, nmallocs_g);
			
			//double malloc_throughput = opts.total_nallocs() / t_malloc * 1e-6;
			//double free_throughput = opts.total_nallocs() / t_free * 1e-6;
			double nallocs = opts.total_nallocs() * opts.palloc * opts.pfree / 
				(opts.palloc + opts.pfree);
			double pair_throughput = nallocs / t_pair * 1e-6;
			//double malloc_speed = opts.total_sz() / t_malloc / NBYTES_IN_GIB;
			double pair_speed = nallocs * opts.expected_sz() / t_pair / 
				NBYTES_IN_GIB;
			//printf("malloc throughput %.2lf Mmallocs/s\n", malloc_throughput);
			//printf("free throughput %.2lf Mfrees/s\n", free_throughput);
			//printf("total test time %.2lf ms\n", t_pair * 1e3);
			printf("pair throughput %.2lf Mpairs/s\n", pair_throughput);
			//printf("malloc speed %.2lf GiB/s\n", malloc_speed);
			printf("pair speed %.2lf GiB/s\n", pair_speed);
		}  // output latency infos

		// free memory
		cucheck(cudaFree(d_ptrs));
		cucheck(cudaFree(d_ctrs));
	}  // operator()
 
};  // ProbThroughputTest

int main(int argc, char **argv) {
	CommonOpts opts;
	run_test<ProbThroughputTest>(argc, argv, opts);
	return 0;
}  // main
