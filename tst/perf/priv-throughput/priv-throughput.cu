/** @file throughput.cu throughput test for various memory allocators */

#include <common.h>

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** measures malloc throughput */
template<class T>
__global__ void throughput_malloc_free_k
(CommonOpts opts, void **ptrs) {
	int n = opts.nthreads, i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	for(int iter = 0; iter < opts.niters; iter++) {
		// first allocate
		for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) 
			ptrs[i + n * ialloc] = T::malloc(opts.next_alloc_sz());
		// then free
		for(int ialloc = 0; ialloc < opts.nallocs; ialloc++) 
			T::free(ptrs[i + n * ialloc]);
	}
}  // throughput_malloc_k

template<class T> class PrivThroughputTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		// allocate memory
		if(warmup) {
			opts.nthreads = min(4 * opts.bs, opts.nthreads);
			opts.ntries = 1;
		}
		if(!warmup)
			printf("private throughput test\n");
		int n = opts.nthreads, bs = opts.bs, grid = divup(n, bs);
		int nptrs = n * opts.nallocs;
		size_t ptrs_sz = nptrs * sizeof(void *);
		void **d_ptrs;
		cucheck(cudaMalloc((void **)&d_ptrs, ptrs_sz));
		cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));

		double t_pair = 0;

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			// allocate
			double t_pair_start = omp_get_wtime();
			throughput_malloc_free_k<T> <<<grid, bs>>>(opts, d_ptrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			double t_pair_end = omp_get_wtime();
			t_pair += t_pair_end - t_pair_start;
			// as pointers have not been zeroed out, check them nevertheless
			if(!check_nz(d_ptrs, 0, nptrs, opts)) {
				fprintf(stderr, "cannot allocate enough memory\n");
				exit(-1);
			}
		}  // for(itry)

		// output throughput infos; no individual malloc/free throughput can be
		// estimated
		if(!warmup) {
			double pair_throughput = opts.total_nallocs() / t_pair * 1e-6;
			double pair_speed = opts.total_sz() / t_pair / NBYTES_IN_GIB;
			printf("pair throughput %.2lf Mpairs/s\n", pair_throughput);
			printf("pair speed %.2lf GiB/s\n", pair_speed);
		}  // output latency infos

		// free memory
		cucheck(cudaFree(d_ptrs));
	}  // operator() 
};  // PrivThroughputTest

int main(int argc, char **argv) {
	CommonOpts opts(true);
	run_test<PrivThroughputTest>(argc, argv, opts);
	return 0;
}  // main
