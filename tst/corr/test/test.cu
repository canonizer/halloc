/** @file test.cu testing a simple idea of an allocator */

#include <halloc.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/sort.h>

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

//#include "halloc.h"

/** testing parameters */
#define NTHREADS (2 * 1024 * 1024)
#define NMALLOCS 8
#define NTHREADS2 (NTHREADS / NMALLOCS)
//#define NTHREADS2 NTHREADS
#define BS 256
#define NTRIES 8
#define MEMORY (4 * 16 * NTHREADS)
//#define NTRIES 1

// alloc/free kernel
__global__ void malloc_free_k(int ntimes) {
	for(int i = 0; i < ntimes; i++) {
		void *p = hamalloc(16);
		if(!p)
			printf("cannot allocate memory\n");
		hafree(p);
	}
}  // malloc_free_k

// alloc-and-save-pointer kernel
__global__ void malloc_k(void **ptrs, int ntimes) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int nthreads = blockDim.x * gridDim.x;
	for(int iptr = tid; iptr < ntimes * nthreads; iptr += nthreads) {
		ptrs[iptr] = hamalloc(16);
		if(!ptrs[iptr])
			printf("cannot allocate memory\n");
	}
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
	for(int iptr = tid; iptr < ntimes * nthreads; iptr += nthreads) {
		ptrs[iptr] = malloc(16);
		if(!ptrs[iptr])
			printf("cannot allocate memory using CUDA malloc()\n");
	}
}  // malloc_k
// read-and-free pointer kernel
__global__ void cuda_free_k(void **ptrs, int ntimes) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int nthreads = blockDim.x * gridDim.x;
	for(int iptr = tid; iptr < ntimes * nthreads; iptr += nthreads)
		free(ptrs[iptr]);
}  // free_k

// a kernel to check whether pointers are good
__global__ void check_ptrs_k(bool *good, uint sz, size_t *ptrs, uint n) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t *ptr = (size_t *)ptrs[i];
	// check 1: try to write two values at the pointer
	ptr[0] = ptrs[i];
	ptr[1] = ptrs[i];
	// check 2: check that the pointer addresses are really valid
	if(i < n - 1) {
		good[i] = ptrs[i + 1] - ptrs[i] >= sz;
	} else
		good[i] = true;
}  // check_ptrs_k

// correctness test - checks if all allocations are correct
void run_test0(void) {
	void **d_ptrs;
	size_t ptrs_sz = NTHREADS2 * NMALLOCS * sizeof(void *);
	uint nmallocs = NMALLOCS * NTHREADS2;
	cucheck(cudaMalloc(&d_ptrs, ptrs_sz));
	size_t *d_addresses = (size_t *)d_ptrs;
	cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));
	// allocate data
	malloc_k<<<NTHREADS2 / BS, BS>>>(d_ptrs, NMALLOCS);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	// sort pointers
	thrust::device_ptr<size_t> dt_addresses(d_addresses);
	thrust::sort(dt_addresses, dt_addresses + nmallocs);
	// check sorted pointers
	bool *d_good;
	size_t good_sz = nmallocs * sizeof(bool);
	cucheck(cudaMalloc((void **)&d_good, good_sz));
	check_ptrs_k<<<nmallocs/BS, BS>>>(d_good, 16, d_addresses, nmallocs);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	thrust::device_ptr<bool> dt_good(d_good);
	bool passed = thrust::all_of(dt_good, dt_good + nmallocs, 
															 thrust::identity<bool>());
	printf("test 0 (correctness of allocation):\n");
	printf("test %s\n", passed ? "PASSED" : "FAILED");
	printf("\n");
	// FINISHED HERE
	// TODO: check pointers (each should point to enough memory)
	// free memory
	free_k<<<NTHREADS2 / BS, BS>>>(d_ptrs, NMALLOCS);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	cucheck(cudaFree(d_ptrs));
}  // run_test0

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
	printf("test duration %.2lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("allocation speed: %.2lf Mpairs/s\n", nmallocs / (t2 - t1) * 1e-6);
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
	cucheck(cudaFree(d_ptrs));
	double nmallocs = (double)NMALLOCS * NTHREADS2 * NTRIES;
	printf("test 2 (first all mallocs, then all frees):\n");
	printf("test duration %.2lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("allocation speed: %.2lf Mpairs/s\n", nmallocs / (t2 - t1) * 1e-6);
	printf("\n");
}  // run_test2

/** latency test */
void run_test3(void) {
	double t1 = omp_get_wtime();
	int lat_ntries = 4, lat_nmallocs = 16 * 1024;
	//int lat_ntries = 1, lat_nmallocs = 1;
	for(int itry = 0; itry < lat_ntries; itry++) {
		malloc_free_k<<<1, 1>>>(lat_nmallocs);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t2 = omp_get_wtime();
	double nmallocs = (double)lat_nmallocs * lat_ntries;
	printf("test 3 (latency):\n");
	printf("test duration %.2lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("latency: %.0lf ns\n", (t2 - t1) * 1e9 / nmallocs);
	printf("\n");
}  // run_test3

/** throughput test for CUDA allocator */
void run_test4(void) {
	void **d_ptrs;
	int cuda_nthreads = 128 * 1024, cuda_nmallocs = 2, cuda_ntries = 4;
	//cucheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 32 * 1024 * 1024));
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
	cucheck(cudaFree(d_ptrs));
	double nmallocs = (double)cuda_nmallocs * cuda_nthreads * cuda_ntries;
	printf("test 4 (CUDA, first all mallocs, then all frees):\n");
	printf("test duration %.2lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("allocation speed: %.2lf Mpairs/s\n", nmallocs / (t2 - t1) * 1e-6);
	printf("\n");
}  // run_test4

// separate time, first for allocation, then for free
void run_test5(void) {
	void **d_ptrs;
	size_t ptrs_sz = NTHREADS2 * NMALLOCS * sizeof(void *);
	cucheck(cudaMalloc(&d_ptrs, ptrs_sz));
	cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));
	uint ntries = 1;
	double t1 = omp_get_wtime();
	for(int itry = 0; itry < ntries; itry++) {
		malloc_k<<<NTHREADS2 / BS, BS>>>(d_ptrs, NMALLOCS);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t2 = omp_get_wtime();
	for(int itry = 0; itry < ntries; itry++) {
		free_k<<<NTHREADS2 / BS, BS>>>(d_ptrs, NMALLOCS);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t3 = omp_get_wtime();
	cucheck(cudaFree(d_ptrs));
	double nmallocs = (double)NMALLOCS * NTHREADS2 * ntries;
	printf("test 5 (first mallocs, then frees, separate timing):\n");
	printf("test duration: malloc %.2lf ms, free %.2lf ms\n", 
				 (t2 - t1) * 1e3, (t3 - t2) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("speed: %.2lf Mmallocs/s, %.2lf Mfrees/s\n", 
				 nmallocs / (t2 - t1) * 1e-6, nmallocs / (t3 - t2) * 1e-6);
	printf("\n");
}  // run_test5

int main(int argc, char **argv) {
	ha_init(halloc_opts_t(MEMORY));
	//ha_init(halloc_opts_t(1024 * 1024 * 1024));
	run_test0();
	run_test1();
	run_test2();
	run_test3();
	run_test4();
	run_test5();
	ha_shutdown();
}  // main
