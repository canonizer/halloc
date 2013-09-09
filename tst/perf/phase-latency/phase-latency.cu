/** @file prob-throughput.cu probabalitized throughput test */

#include <common.h>

#include <float.h>
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
__global__ void phase_latency_k
(void **ptrs, uint *ctrs, uint itry, ActionType *actions, uint *latencies) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	uint n = opts_g.nthreads;
	uint nallocs = opts_g.nallocs;
	if(opts_g.is_thread_inactive(i))
		return;
	uint ctr = ctrs[i];
	//uint nmallocs = 0;

	// iterate 
	for(uint iter = 0; iter < opts_g.niters; iter++) {
		// perform the action
		ActionType action = opts_g.next_action(ctr > 0, itry, iter);
		actions[iter * n + i] = action;
		switch(action) {
			//switch(ctr > 0 ? ActionFree : ActionAlloc) {
		case ActionAlloc:
			for(uint ialloc = 0; ialloc < opts_g.nallocs; ialloc++) {
				uint t1 = clock();
				ptrs[ialloc * n + i] = T::malloc(opts_g.next_alloc_sz());
				uint t2 = clock();
				latencies[(iter * nallocs + ialloc) * n + i] = t2 - t1;
			}
			ctr = opts_g.nallocs;
			//nmallocs += nallocs;
			break;
		case ActionFree:
			for(uint ialloc = 0; ialloc < opts_g.nallocs; ialloc++) {
				uint t1 = clock();
				T::free(ptrs[ialloc * n + i]);
				uint t2 = clock();
				latencies[(iter * nallocs + ialloc) * n + i] = t2 - t1;
			}
			ctr = 0;
			break;
		case ActionNone:
			//printf("no action taken\n");
			break;
		}
	}  // for(each iteration)
	ctrs[i] = ctr;
}  // phase_latency_k

/** measures malloc/free latency */
template<class T> class PhaseLatencyTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		// allocate memory
		if(warmup) {
			opts.nthreads = min(4 * opts.bs, opts.nthreads);
			opts.ntries = 1;
			opts.niters = 1;
		}
		if(!warmup)
			printf("two-phase latency test\n");
		cuset(opts_g, CommonOpts, opts);
		int n = opts.nthreads, bs = opts.bs, grid = divup(n, bs);
		int nptrs = n * opts.nallocs;
		size_t ptrs_sz = nptrs * sizeof(void *);
		uint ctrs_sz = n * sizeof(uint);
		size_t lat_sz = n * opts.niters * opts.nallocs * sizeof(uint);
		size_t act_sz = n * opts.niters * sizeof(ActionType);
		void **d_ptrs;
		cucheck(cudaMalloc((void **)&d_ptrs, ptrs_sz));
		cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));
		uint *d_ctrs;
		cucheck(cudaMalloc((void **)&d_ctrs, ctrs_sz));
		cucheck(cudaMemset(d_ctrs, 0, ctrs_sz));
		uint *d_latencies;
		cucheck(cudaMalloc((void **)&d_latencies, lat_sz));
		cucheck(cudaMemset(d_latencies, 0, lat_sz));
		ActionType *d_actions;
		cucheck(cudaMalloc((void **)&d_actions, act_sz));
		cucheck(cudaMemset(d_actions, 0, act_sz));
		uint *h_latencies;
		cucheck(cudaMallocHost((void **)&h_latencies, lat_sz));
		ActionType *h_actions;
		cucheck(cudaMallocHost((void **)&h_actions, act_sz));
		
		//cuset(nmallocs_g, uint64, 0);

		// latency variables
		double avg_malloc_latency = 0, avg_free_latency = 0;
		double min_malloc_latency = FLT_MAX, min_free_latency = FLT_MAX;
		double max_malloc_latency = FLT_MIN, max_free_latency = FLT_MIN;
		double nmallocs = 0, nfrees = 0;

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			// run the kernel
			phase_latency_k<T> <<<grid, bs>>>
				(d_ptrs, d_ctrs, itry, d_actions, d_latencies);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			// check that pointers are correct
			if(!check_nz(d_ptrs, d_ctrs, nptrs, opts)) {
				fprintf(stderr, "cannot allocate enough memory\n");
				exit(-1);
			}
			// compute the latencies
			if(!warmup) {
				cucheck(cudaMemcpy(h_latencies, d_latencies, lat_sz,
													 cudaMemcpyDeviceToHost));
				cucheck(cudaMemcpy(h_actions, d_actions, act_sz,
													 cudaMemcpyDeviceToHost));
				for(int iter = 0; iter < opts.niters; iter++) {
					for(int i = 0; i < n; i += opts.period_mask + 1) {
						ActionType action = h_actions[iter * n + i];
						for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) {
							uint latency = 
								h_latencies[(iter * opts.nallocs + ialloc) * n  + i];
							//double malloc_latency = h_malloc_latencies[ialloc * n + i];
							//double free_latency = h_free_latencies[ialloc * n + i];
							switch(action) {
							case ActionAlloc:
								nmallocs++;
								avg_malloc_latency += (double)latency;
								min_malloc_latency = min(min_malloc_latency, (double)latency);
								max_malloc_latency = max(max_malloc_latency, (double)latency);
								break;
							case ActionFree:
								nfrees++;
								avg_free_latency += (double)latency;
								min_free_latency = min(min_free_latency, (double)latency);
								max_free_latency = max(max_free_latency, (double)latency);
								break;
								// otherwise, do nothing
							}
						}  // for(ialloc)
					}  // for(i)
				}  // for(iter)
			}  // if(not warmup)

		}  // for(itry)

		// free the rest - this is not timed for latency
		{
			free_rest_k<T> <<<grid, bs>>> (/* opts, */ d_ptrs, d_ctrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
		}

		// output throughput infos
		if(!warmup) {
			avg_malloc_latency /= nmallocs;
			avg_free_latency /= nfrees;
			printf("min malloc latency %.2lf cycles\n", min_malloc_latency);
			printf("avg malloc latency %.2lf cycles\n", avg_malloc_latency);
			printf("max malloc latency %.2lf cycles\n", max_malloc_latency);
			printf("min free latency %.2lf cycles\n", min_free_latency);
			printf("avg free latency %.2lf cycles\n", avg_free_latency);
			printf("max free latency %.2lf cycles\n", max_free_latency);
			printf("avg pair latency %.2lf cycles\n", 
						 avg_malloc_latency + avg_free_latency);
		}  // output latency infos

		// free memory
		cucheck(cudaFree(d_ptrs));
		cucheck(cudaFree(d_ctrs));
	}  // operator() 
};  // PhaseLatencyTest

int main(int argc, char **argv) {
	CommonOpts opts(true);
	run_test<PhaseLatencyTest>(argc, argv, opts);
	return 0;
}  // main
