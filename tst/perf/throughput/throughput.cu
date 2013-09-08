/** @file throughput.cu throughput test for various memory allocators */

#include <common.h>

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** measures malloc throughput */

template<class T> class ThroughputTest {
	
public:
	void operator()(CommonOpts opts, bool warmup) {
		opts.niters = 1;
		// allocate memory
		if(warmup) {
			opts.nthreads = min(4 * opts.bs, opts.nthreads);
			opts.ntries = 1;
		}
		if(!warmup)
			printf("throughput test\n");
		int n = opts.nthreads, bs = opts.bs, grid = divup(n, bs);
		int nptrs = n * opts.nallocs;
		size_t ptrs_sz = nptrs * sizeof(void *);
		void **d_ptrs;
		cucheck(cudaMalloc((void **)&d_ptrs, ptrs_sz));
		cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));

		double t_malloc = 0, t_free = 0;

		// do testing
		for(int itry = 0; itry < opts.ntries; itry++) {
			// allocate
			double t_malloc_start = omp_get_wtime();
			malloc_k<T> <<<grid, bs>>>(opts, d_ptrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			double t_malloc_end = omp_get_wtime();
			t_malloc += t_malloc_end - t_malloc_start;
			// check that pointers are correct
			if(!check_nz(d_ptrs, 0, nptrs, opts)) {
				fprintf(stderr, "cannot allocate enough memory\n");
				exit(-1);
			}
			// free
			double t_free_start = omp_get_wtime();
			free_k<T> <<<grid, bs>>>(opts, d_ptrs);
			cucheck(cudaGetLastError());
			cucheck(cudaStreamSynchronize(0));
			double t_free_end = omp_get_wtime();
			t_free += t_free_end - t_free_start;
		}  // for(itry)

		// output latency infos
		if(!warmup) {
			double malloc_throughput = opts.total_nallocs() / t_malloc * 1e-6;
			double free_throughput = opts.total_nallocs() / t_free * 1e-6;
			double pair_throughput = opts.total_nallocs() / (t_malloc + t_free) 
				* 1e-6;
			double malloc_speed = opts.total_sz() / t_malloc / NBYTES_IN_GIB;
			double pair_speed = opts.total_sz() / (t_malloc + t_free) / NBYTES_IN_GIB;
			printf("malloc throughput %.2lf Mmallocs/s\n", malloc_throughput);
			printf("free throughput %.2lf Mfrees/s\n", free_throughput);
			printf("pair throughput %.2lf Mpairs/s\n", pair_throughput);
			printf("malloc speed %.2lf GiB/s\n", malloc_speed);
			printf("pair speed %.2lf GiB/s\n", pair_speed);
		}  // output latency infos

		// free memory
		cucheck(cudaFree(d_ptrs));		
	}  // operator()
 
};  // LatencyTest

int main(int argc, char **argv) {
	CommonOpts opts(true);
	run_test<ThroughputTest>(argc, argv, opts);
	return 0;
}  // main
