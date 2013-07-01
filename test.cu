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


/** memory allocation; @nbytes is ignored */
__device__ void *hamalloc(size_t nbytes);

/** freeing the memory */
__device__ void hafree(void *p);

/** initializes memory allocator host-side */
void ha_init(void);

/** shuts down memory allocator host-side */
void ha_shutdown(void);

/** allocator parameters */
#define BLOCK_SZ 16
#define NBLOCKS (16 * 1024 * 1024)
#define HASH_STEP (NBLOCKS / 128 + NBLOCKS / 32 - 1)
#define NCOUNTERS 2048

/** testing parameters */
#define NTHREADS (4 * 1024 * 1024)
#define NMALLOCS 1
#define BS 512
#define NTRIES 16

/** the buffer from which to allocate memory, [nblocks_g * block_sz_g] bytes */
__device__ void *blocks_g;
/** total number of blocks; always a power of two */
__device__ uint nblocks_g = NBLOCKS;
/** size of a single block in bytes (16 bytes by default) */
__device__ uint block_sz_g = BLOCK_SZ;
/** the step used for linear hash function */
__device__ uint hash_step_g = HASH_STEP;
/** bits indicating block occupancy (0 = free block, 1 = occupied block),
		[nblocks_g / (sizeof(uint) * 8)] */
__device__ uint *block_bits_g;
/** total number of counters (a power of 2) */
__device__ uint ncounters_g = NCOUNTERS;
/** the counters to generate allocation ids, initially set to 0 */
__device__ uint *counters_g;

__device__ void *hamalloc(size_t nbytes) {
	// the counter is based on block id
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint icounter = tid & (ncounters_g - 1);
	uint counter_val = atomicAdd(counters_g + icounter, 1);
	// initial position
	uint iblock = (tid + counter_val * hash_step_g) & (nblocks_g - 1);
	// iterate until successfully reserved
	for(uint i = 0; i < nblocks_g; i++) {
		// try reserve
		uint iword = iblock / 32, ibit = iblock % 32;
		uint old_word = atomicOr(block_bits_g + iword, 1 << ibit);
		if(!((old_word >> ibit) & 1)) {
			// reservation successful, return pointer
			return (char *)blocks_g + iblock * block_sz_g;
		} else 
			iblock = (iblock + hash_step_g) & (nblocks_g - 1);
	}
	// if we are here, then all memory is used
	printf("cannot allocate memory\n");
	return 0;
}  // hamalloc

__device__ void hafree(void *p) {
	// ignore zero pointer
	if(!p)
		return;
	uint iblock = ((char *)p - (char *)blocks_g) / block_sz_g;
	uint iword = iblock / 32, ibit = iblock % 32;
	atomicAnd(block_bits_g + iword, ~(1 << ibit));
}  // hafree

void ha_init(void) {
	// allocate memory
	uint *block_bits, *counters;
	void *blocks;
	cucheck(cudaMalloc(&blocks, NBLOCKS * BLOCK_SZ));
	cucheck(cudaMalloc((void **)&block_bits, NBLOCKS / 32 * sizeof(uint)));
	cucheck(cudaMalloc((void **)&counters, NCOUNTERS * sizeof(uint)));
	// initialize to zero
	cucheck(cudaMemset(block_bits, 0, NBLOCKS / 32 * sizeof(uint)));
	cucheck(cudaMemset(counters, 0, NCOUNTERS * sizeof(uint)));
	// set device-side variables
	void *block_bits_addr, *counters_addr, *blocks_addr;
	cucheck(cudaGetSymbolAddress(&block_bits_addr, block_bits_g));
	cucheck(cudaGetSymbolAddress(&counters_addr, counters_g));
	cucheck(cudaGetSymbolAddress(&blocks_addr, blocks_g));
	cucheck(cudaMemcpy(block_bits_addr, &block_bits, sizeof(void *), 
										 cudaMemcpyHostToDevice));
	cucheck(cudaMemcpy(counters_addr, &counters, sizeof(void *), 
										 cudaMemcpyHostToDevice));
	cucheck(cudaMemcpy(blocks_addr, &blocks, sizeof(void *), 
										 cudaMemcpyHostToDevice));
}  // ha_init

void ha_shutdown(void) {
	// TODO: free memory
}  // ha_shutdown

// alloc/free kernel
__global__ void malloc_free_k(int n) {
	void *p = hamalloc(16);
	hafree(p);
}  // malloc_free_k

// alloc--and-save-pointer kernel
__global__ void malloc_k(void **ptrs) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	ptrs[tid] = hamalloc(16);
}  // malloc_k
// read-and-free pointer kernel
__global__ void free_k(void **ptrs) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	hafree(ptrs[tid]);
}  // free_k

void run_test1(void) {
	double t1 = omp_get_wtime();
	for(int itry = 0; itry < NTRIES; itry++) {
		malloc_free_k<<<NTHREADS / BS, BS>>>(0);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t2 = omp_get_wtime();
	double nmallocs = (double)NMALLOCS * NTHREADS * NTRIES;
	printf("test 1 (malloc/free inside each thread)\n:");
	printf("test duration %.3lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("allocation speed: %.3lf Mpairs/s\n", nmallocs / (t2 - t1) * 1e-6);
	printf("\n");
}  // run_test1

void run_test2(void) {
	void **d_ptrs;
	size_t ptrs_sz = NTHREADS * NMALLOCS * sizeof(void *);
	cucheck(cudaMalloc(&d_ptrs, ptrs_sz));
	cucheck(cudaMemset(d_ptrs, 0, ptrs_sz));
	double t1 = omp_get_wtime();
	for(int itry = 0; itry < NTRIES; itry++) {
		malloc_k<<<NTHREADS / BS, BS>>>(d_ptrs);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
		free_k<<<NTHREADS / BS, BS>>>(d_ptrs);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t2 = omp_get_wtime();
	// copy back and print some allocated pointers
	void **h_ptrs = (void **)malloc(ptrs_sz);
	cucheck(cudaMemcpy(h_ptrs, d_ptrs, ptrs_sz, cudaMemcpyDeviceToHost));
	for(int ip = 0; ip < 256; ip += 7)
		printf("d_ptrs[%d] = %p\n", ip, h_ptrs[ip]);
	double nmallocs = (double)NMALLOCS * NTHREADS * NTRIES;
	printf("test 2 (first all mallocs, then all frees)\n:");
	printf("test duration %.3lf ms\n", (t2 - t1) * 1e3);
	printf("%.0lf malloc/free pairs in the test\n", nmallocs);
	printf("allocation speed: %.3lf Mpairs/s\n", nmallocs / (t2 - t1) * 1e-6);
	printf("\n");
}  // run_test2

int main(int argc, char **argv) {
	ha_init();
	run_test1();
	run_test2();
	ha_shutdown();
}  // main
