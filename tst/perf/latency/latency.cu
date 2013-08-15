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
__global__ void latency_malloc_k
(CommonOpts opts, void **ptrs, double *latencies) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= n || i & opts.period_mask)
		return;
	uint64 t1 = clock64();
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) 
		ptrs[i + n * ialloc] = T::malloc(opts.alloc_sz);
	uint64 t2 = clock64(), latency = t2 - t1;
	latencies[i] += (double)latency;
}  // latency_malloc_k

// TODO: verify that all pointers are non-zero

template<class T>
__global__ void latency_free_k
(CommonOpts opts, void **ptrs, double *latencies) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= n || i & opts.period_mask)
		return;
	uint64 t1 = clock64();
	for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) 
		T::free(ptrs[i + n * ialloc]);
	uint64 t2 = clock64(), latency = t2 - t1;
	latencies[i] += (double)latency;
}  // latency_free_k

template<class T> class LatencyTest {
public:
	void operator()(CommonOpts &opts) {
		// allocate memory
		printf("latency test\n");
		int n = opts.nthreads, bs = opts.bs, grid = divup(n, bs);
		size_t ptrs_sz = n * opts.nallocs * sizeof(void *); 
		size_t lat_sz = n * sizeof(double);
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

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			// allocate
			latency_malloc_k<T> <<<grid, bs>>>(opts, d_ptrs, d_malloc_latencies);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			// free
			latency_free_k<T> <<<grid, bs>>>(opts, d_ptrs, d_free_latencies);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
		}  // for(itry)

		// output latency infos
		cucheck(cudaMemcpy(h_malloc_latencies, d_malloc_latencies, lat_sz,
											 cudaMemcpyDeviceToHost));
		cucheck(cudaMemcpy(h_free_latencies, d_free_latencies, lat_sz,
											 cudaMemcpyDeviceToHost));
		double malloc_latency = 0, free_latency = 0;
		for(int i = 0; i < n; i++) {
			malloc_latency += h_malloc_latencies[i];
			free_latency += h_free_latencies[i];
		}
		malloc_latency /= opts.total_nallocs();
		free_latency /= opts.total_nallocs();
		printf("avg malloc latency %lf cycles\n", malloc_latency);
		printf("avg free latency %lf cycles\n", free_latency);
		printf("avg pair latency %lf cycles\n", malloc_latency + free_latency);

		// free memory
		free(h_malloc_latencies);
		free(h_free_latencies);
		cucheck(cudaFree(d_malloc_latencies));
		cucheck(cudaFree(d_free_latencies));
		cucheck(cudaFree(d_ptrs));
		
	}  // operator() 
};  // LatencyTest

int main(int argc, char **argv) {
	CommonOpts opts;
	run_test<LatencyTest> (argc, argv, opts);
	return 0;
}  // main
