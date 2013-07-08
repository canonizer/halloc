/** @file test.cu testing a simple idea of an allocator */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** a macro for checking CUDA calls */
#define cucheck(call)																										\
	{																																			\
	cudaError_t cucheck_err = (call);																			\
	if(cucheck_err != cudaSuccess) {																			\
		const char* err_str = cudaGetErrorString(cucheck_err);							\
		fprintf(stderr, "%s (%d): %s in %s\n", __FILE__, __LINE__, err_str, #call);	\
		exit(-1);																														\
	}																																			\
	}

#include "halloc.h"

/** testing parameters */
#define NTHREADS (16384)
//#define NTHREADS 16384
#define NMALLOCS 16
#define NTHREADS2 (NTHREADS / NMALLOCS)
#define BS 512
#define NTRIES 1
//#define NTRIES 1

// alloc/free kernel
__global__ void malloc_free_k(int ntimes) {
	for(int i = 0; i < ntimes; i++) {
		void *p = hamalloc(16);
		hafree(p);
	}
}  // malloc_free_k

// alloc-and-save-pointer kernel
__global__ void malloc_k(void **ptrs, int ntimes) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int nthreads = blockDim.x * gridDim.x;
	for(int iptr = tid; iptr < ntimes * nthreads; iptr += nthreads)
		ptrs[iptr] = hamalloc(16);
}  // malloc_k
// read-and-free pointer kernel
__global__ void free_k(void **ptrs, int ntimes) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int nthreads = blockDim.x * gridDim.x;
	for(int iptr = tid; iptr < ntimes * nthreads; iptr += nthreads)
		hafree(ptrs[iptr]);
}  // free_k

// alloc-and-save-pointer kernel
__global__ void cuda_malloc_k(void **ptrs, int ntimes) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int nthreads = blockDim.x * gridDim.x;
	for(int iptr = tid; iptr < ntimes * nthreads; iptr += nthreads)
		ptrs[iptr] = malloc(16);
}  // malloc_k
// read-and-free pointer kernel
__global__ void cuda_free_k(void **ptrs, int ntimes) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int nthreads = blockDim.x * gridDim.x;
	for(int iptr = tid; iptr < ntimes * nthreads; iptr += nthreads)
		free(ptrs[iptr]);
}  // free_k

void run_test1(void) {
	double t1 = omp_get_wtime();
	for(int itry = 0; itry < NTRIES; itry++) {
		malloc_free_k<<<NTHREADS / BS, BS>>>(1);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t2 = omp_get_wtime();
	double nmallocs = (double)NTHREADS * NTRIES;
	printf("test 1 (malloc/free inside each thread):\n");
	printf("test duration %.3lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("allocation speed: %.3lf Mpairs/s\n", nmallocs / (t2 - t1) * 1e-6);
	printf("\n");
}  // run_test1

void run_test2(void) {
	void **d_ptrs;
	size_t ptrs_sz = NTHREADS2 * NMALLOCS * sizeof(void *);
	cucheck(cudaMalloc(&d_ptrs, ptrs_sz));
	cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));
	double t1 = omp_get_wtime();
	for(int itry = 0; itry < NTRIES; itry++) {
		malloc_k<<<NTHREADS2 / BS, BS>>>(d_ptrs, NMALLOCS);
		cucheck(cudaGetLastError());
		//cucheck(cudaStreamSynchronize(0));
		free_k<<<NTHREADS2 / BS, BS>>>(d_ptrs, NMALLOCS);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t2 = omp_get_wtime();
	// copy back and print some allocated pointers
	void **h_ptrs = (void **)malloc(ptrs_sz);
	cucheck(cudaMemcpy(h_ptrs, d_ptrs, ptrs_sz, cudaMemcpyDeviceToHost));
	cucheck(cudaFree(d_ptrs));
	//for(int ip = 0; ip < 256; ip += 7)
	//	printf("d_ptrs[%d] = %p\n", ip, h_ptrs[ip]);
	free(h_ptrs);
	double nmallocs = (double)NMALLOCS * NTHREADS2 * NTRIES;
	printf("test 2 (first all mallocs, then all frees):\n");
	printf("test duration %.3lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("allocation speed: %.3lf Mpairs/s\n", nmallocs / (t2 - t1) * 1e-6);
	printf("\n");
}  // run_test2

/** latency test */
void run_test3(void) {
	double t1 = omp_get_wtime();
	int lat_ntries = 8, lat_nmallocs = 16 * 1024;
	for(int itry = 0; itry < lat_ntries; itry++) {
		malloc_free_k<<<1, 1>>>(lat_nmallocs);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t2 = omp_get_wtime();
	double nmallocs = (double)lat_nmallocs * lat_ntries;
	printf("test 3 (latency):\n");
	printf("test duration %.3lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("latency: %.0lf ns\n", (t2 - t1) * 1e9 / nmallocs);
	printf("\n");
}  // run_test3

/** throughput test for CUDA allocator */
void run_test4(void) {
	void **d_ptrs;
	int cuda_nthreads = 32768, cuda_nmallocs = 2, cuda_ntries = 4;
	size_t ptrs_sz = cuda_nthreads * cuda_nmallocs * sizeof(void *);
	cucheck(cudaMalloc(&d_ptrs, ptrs_sz));
	cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));
	double t1 = omp_get_wtime();
	for(int itry = 0; itry < cuda_ntries; itry++) {
		cuda_malloc_k<<<cuda_nthreads / BS, BS>>>(d_ptrs, cuda_nmallocs);
		cucheck(cudaGetLastError());
		//cucheck(cudaStreamSynchronize(0));
		cuda_free_k<<<cuda_nthreads / BS, BS>>>(d_ptrs, cuda_nmallocs);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t2 = omp_get_wtime();
	// copy back and print some allocated pointers
	void **h_ptrs = (void **)malloc(ptrs_sz);
	cucheck(cudaMemcpy(h_ptrs, d_ptrs, ptrs_sz, cudaMemcpyDeviceToHost));
	cucheck(cudaFree(d_ptrs));
	//for(int ip = 0; ip < 256; ip += 7)
	//	printf("d_ptrs[%d] = %p\n", ip, h_ptrs[ip]);
	free(h_ptrs);
	double nmallocs = (double)cuda_nmallocs * cuda_nthreads * cuda_ntries;
	printf("test 4 (CUDA, first all mallocs, then all frees):\n");
	printf("test duration %.3lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("allocation speed: %.3lf Mpairs/s\n", nmallocs / (t2 - t1) * 1e-6);
	printf("\n");
}  // run_test4

int main(int argc, char **argv) {
	ha_init();
	run_test1();
	run_test2();
	run_test3();
	run_test4();
	ha_shutdown();
}  // main
