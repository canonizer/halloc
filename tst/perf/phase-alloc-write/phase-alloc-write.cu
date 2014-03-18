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
			for(uint ialloc = 0; ialloc < opts_g.nallocs; ialloc++) {
				// allocate
				uint alloc_sz = opts_g.next_alloc_sz();
				uint64 *p =	(uint64 *)T::malloc(alloc_sz);
				for(int iword = 0; iword < alloc_sz / (uint)sizeof(uint64); iword++) 
					p[iword] = 123ull;
				ptrs[ialloc * n + i] = p;
			}
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
template<class T> class PhaseThroughputTest {
	
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
		
		double t_malloc = 0, t_free = 0, t_pair = 0;
		cuset(nmallocs_g, uint64, 0);

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			// run the kernel
			double t_start = omp_get_wtime();
			prob_throughput_k<T> <<<grid, bs>>>(/* opts, */ d_ptrs, d_ctrs, itry);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			double t_end = omp_get_wtime(), dt = t_end - t_start;
			t_pair += dt;
			if(itry % 2)
				t_free += dt;
			else
				t_malloc += dt;
			// check that pointers are correct
			if(!check_nz(d_ptrs, d_ctrs, nptrs, opts)) {
				fprintf(stderr, "cannot allocate enough memory\n");
				exit(-1);
			}		
		}  // for(itry)

		// free the rest
		{
			double t_start = omp_get_wtime();
			free_rest_k<T> <<<grid, bs>>> (/* opts, */ d_ptrs, d_ctrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			double t_end = omp_get_wtime(), dt = t_end - t_start;
			t_pair += dt;
			t_free += dt;
		}

		// output throughput infos
		if(!warmup) {
			//uint64 nallocs;
			//cuget(&nallocs, nmallocs_g);
			
			//double malloc_throughput = opts.total_nallocs() / t_malloc * 1e-6;
			//double free_throughput = opts.total_nallocs() / t_free * 1e-6;
			double npairs = 0.5 * opts.total_nallocs() * opts.exec_fraction;
			double nmallocs = 0.25 * opts.total_nallocs() * 
				(opts.exec_fraction + opts.alloc_fraction - opts.free_fraction);
			double nfrees = 0.25 * opts.total_nallocs() * 
				(opts.exec_fraction + opts.alloc_fraction - opts.free_fraction);
			double pair_throughput = npairs / t_pair * 1e-6;
			double malloc_throughput = nmallocs / t_malloc * 1e-6;
			double free_throughput = nfrees / t_free * 1e-6;
			double malloc_speed = nmallocs * opts.expected_sz() / t_malloc /
				NBYTES_IN_GIB;
			double pair_speed = npairs * opts.expected_sz() / t_pair /
				NBYTES_IN_GIB;
			if(opts.niters == 1) {
				printf("malloc throughput %.2lf Mmallocs/s\n", malloc_throughput);
				printf("free throughput %.2lf Mfrees/s\n", free_throughput);
			}
			//printf("total test time %.2lf ms\n", t_pair * 1e3);
			printf("pair throughput %.2lf Mpairs/s\n", pair_throughput);
			if(opts.niters == 1) {
				printf("malloc speed %.2lf GiB/s\n", malloc_speed);
			}
			printf("pair speed %.2lf GiB/s\n", pair_speed);
		}  // output latency infos

		// free memory
		cucheck(cudaFree(d_ptrs));
		cucheck(cudaFree(d_ctrs));
	}  // operator()
 
};  // PhaseThroughputTest

int main(int argc, char **argv) {
	CommonOpts opts(true);
	run_test<PhaseThroughputTest>(argc, argv, opts);
	return 0;
}  // main
