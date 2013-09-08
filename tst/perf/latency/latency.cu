/** @file latency.cu latency test for various memory allocators */

#include <common.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** measures malloc latencies; note that latencies are averaged per-thread,
		per-allocation latencies are not preserved; latencies here are measured in cycles */
template<class T>
__global__ void latency_malloc_k
(CommonOpts opts, void **ptrs, double *latencies) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) {
		uint sz = opts.next_alloc_sz();
		uint64 t1 = clock64();
		ptrs[i + n * ialloc] = T::malloc(sz);
		uint64 t2 = clock64(), latency = t2 - t1;
		latencies[i + ialloc * n] = (double)latency;
	}
}  // latency_malloc_k

// TODO: verify that all pointers are non-zero

template<class T>
__global__ void latency_free_k
(CommonOpts opts, void **ptrs, double *latencies) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) {
		uint64 t1 = clock64();
		T::free(ptrs[i + n * ialloc]);
		uint64 t2 = clock64(), latency = t2 - t1;
		latencies[i + ialloc * n] = (double)latency;
	}
}  // latency_free_k

template<class T> class LatencyTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		opts.niters = 1;
		// allocate memory
		if(warmup) {
			opts.nthreads = min(4 * opts.bs, opts.nthreads);
			opts.ntries = 1;
		}
		if(!warmup)
			printf("latency test\n");
		int n = opts.nthreads, bs = opts.bs, grid = divup(n, bs);
		int nptrs = n * opts.nallocs;
		size_t ptrs_sz = nptrs * sizeof(void *);
		size_t lat_sz = nptrs * sizeof(double);
		void **d_ptrs;
		cucheck(cudaMalloc((void **)&d_ptrs, ptrs_sz));
		cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));
		double *h_malloc_latencies = (double *)malloc(lat_sz);
		double *h_free_latencies = (double *)malloc(lat_sz);
		double *d_malloc_latencies, *d_free_latencies;
		cucheck(cudaMalloc((void **)&d_malloc_latencies, lat_sz));
		cucheck(cudaMalloc((void **)&d_free_latencies, lat_sz));
		cucheck(cudaMemset(d_malloc_latencies, 0, lat_sz));
		cucheck(cudaMemset(d_free_latencies, 0, lat_sz));

		// latency variables
		double avg_malloc_latency = 0, avg_free_latency = 0;
		double min_malloc_latency = FLT_MAX, min_free_latency = FLT_MAX;
		double max_malloc_latency = FLT_MIN, max_free_latency = FLT_MIN;

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			// allocate
			latency_malloc_k<T> <<<grid, bs>>>(opts, d_ptrs, d_malloc_latencies);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			// check that pointers are correct
			if(!check_nz(d_ptrs, 0, nptrs, opts)) {
				fprintf(stderr, "cannot allocate enough memory\n");
				exit(-1);
			}
			// free
			latency_free_k<T> <<<grid, bs>>>(opts, d_ptrs, d_free_latencies);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			// collect latency infos
			if(!warmup) {
				cucheck(cudaMemcpy(h_malloc_latencies, d_malloc_latencies, lat_sz,
													 cudaMemcpyDeviceToHost));
				cucheck(cudaMemcpy(h_free_latencies, d_free_latencies, lat_sz,
													 cudaMemcpyDeviceToHost));
				for(int i = 0; i < n; i += opts.period_mask + 1) {
					for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) {					
						double malloc_latency = h_malloc_latencies[ialloc * n + i];
						double free_latency = h_free_latencies[ialloc * n + i];
						avg_malloc_latency += malloc_latency;
						avg_free_latency += free_latency;
						min_malloc_latency = min(min_malloc_latency, malloc_latency);
						min_free_latency = min(min_free_latency, free_latency);
						max_malloc_latency = max(max_malloc_latency, malloc_latency);
						max_free_latency = max(max_free_latency, free_latency);
					}
				}
			}  // if(not warmup)
		}  // for(itry)

		// output latency infos
		if(!warmup) {
			avg_malloc_latency /= opts.total_nallocs();
			avg_free_latency /= opts.total_nallocs();
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
		free(h_malloc_latencies);
		free(h_free_latencies);
		cucheck(cudaFree(d_malloc_latencies));
		cucheck(cudaFree(d_free_latencies));
		cucheck(cudaFree(d_ptrs));		
	}  // operator()
 
};  // LatencyTest

int main(int argc, char **argv) {
	CommonOpts opts(true);
	run_test<LatencyTest> (argc, argv, opts);
	return 0;
}  // main
