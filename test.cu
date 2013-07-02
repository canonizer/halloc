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
#define HASH_STEP (NBLOCKS / 256 + NBLOCKS / 64 - 1)
//#define HASH_STEP (NBLOCKS / (64) - 1)
#define NCOUNTERS 2048
#define THREAD_FREQ 17
#define COUNTER_INC 1
//#define NCOUNTERS 2048

/** testing parameters */
#define NTHREADS (8 * 1024 * 1024)
#define NMALLOCS 16
#define NTHREADS2 (NTHREADS / NMALLOCS)
#define BS 512
#define NTRIES 128
//#define NTRIES 1

/** the buffer from which to allocate memory, [nblocks_g * block_sz_g] bytes */
__device__ void *blocks_g;
/** total number of blocks; always a power of two */
__constant__ uint nblocks_g = NBLOCKS;
/** size of a single block in bytes (16 bytes by default) */
__device__ uint block_sz_g = BLOCK_SZ;
/** the step used for linear hash function */
__constant__ uint hash_step_g = HASH_STEP;
/** bits indicating block occupancy (0 = free block, 1 = occupied block),
		[nblocks_g / (sizeof(uint) * 8)] */
__device__ uint *block_bits_g;
/** total number of counters (a power of 2) */
__constant__ uint ncounters_g = NCOUNTERS;
/** the counters to generate allocation ids, initially set to 0 */
__device__ uint *counters_g;

__device__ void *hamalloc(size_t nbytes) {
	// the counter is based on block id
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint wid = tid / 32, lid = tid % 32;
	uint leader_lid = __ffs(__ballot(1)) - 1;
	//uint icounter = tid & (ncounters_g - 1);
	uint icounter = wid & (NCOUNTERS - 1);
	//uint counter_val = atomicAdd(counters_g + icounter, 1);
	uint counter_val;
	if(lid == leader_lid)
		counter_val = atomicAdd(counters_g + icounter, COUNTER_INC);
	counter_val = __shfl((int)counter_val, leader_lid);
	// initial position
	// TODO: use a real but cheap random number generator
	//uint iblock = (tid + counter_val * hash_step_g) & (nblocks_g - 1);
	uint iblock = (tid * THREAD_FREQ + counter_val * counter_val * (counter_val + 1)) 
		& (nblocks_g - 1);
	// iterate until successfully reserved
	for(uint i = 0; i < nblocks_g; i++) {
	//for(uint i = 0; i < 1; i++) {
		// try reserve
		uint iword = iblock / 32, ibit = iblock % 32;
		uint alloc_mask = 1 << ibit;
		uint old_word = atomicOr(block_bits_g + iword, alloc_mask);
		if(!(old_word & alloc_mask)) {
			// reservation successful, return pointer
			return (char *)blocks_g + iblock * block_sz_g;
		} else 
			iblock = (iblock + hash_step_g) & (nblocks_g - 1);
	}
	// if we are here, then all memory is used
	//printf("cannot allocate memory\n");
	return 0;
}  // hamalloc

#define MAX_NWARPS 32

__device__ uint warp_reduce_or(uint val, uint leader_lid) {
	volatile __shared__ uint buf[MAX_NWARPS];
	uint wid = threadIdx.x / 32, lid = threadIdx.x % 32;
	if(lid == leader_lid)
		buf[wid] = 0;
	atomicOr((uint *)&buf[wid], val);
	return val;
}  // warp_reduce_or

__device__ void hafree(void *p) {
	// ignore zero pointer
	if(!p)
		return;
	// free the memory
	uint iblock = ((char *)p - (char *)blocks_g) / block_sz_g;
	uint iword = iblock / 32, ibit = iblock % 32;
	atomicAnd(block_bits_g + iword, ~(1 << ibit));

	// uint lid = threadIdx.x % 32;
	// // warp-aggregated atomics
	// bool want_free = true;
	// while(__any(want_free)) {
	// 	if(want_free) {
	// 		uint leader_lid = __ffs(__ballot(want_free)) - 1;
	// 		uint leader_iword = __shfl((int)iword, leader_lid);
	// 		uint free_mask = leader_iword == iword ? 1 << ibit : 0;
	// 		// reduce the mask across lanes
	// 		//for(uint i = 16; i >= 1; i /= 2)
	// 		//	free_mask |= __shfl_xor((int)free_mask, i, 32);
	// 		free_mask = warp_reduce_or(free_mask, leader_lid);
	// 		if(leader_lid == lid)
	// 			atomicAnd(block_bits_g + iword, ~free_mask);
	// 		want_free = want_free && leader_iword != iword;
	// 	}  // if(want_free)
	// }  // while(any wants to free)
}  // hafree

/** warp-based malloc implementation */
__device__ void *hamalloc2(size_t nbytes) {
	// TODO: find out where it fails
	// the counter is based on block id
	uint tid = threadIdx.x + blockIdx.x * blockDim.x, wid = tid / 32;
	uint lid = tid % 32;
	uint icounter = wid & (ncounters_g - 1);
	uint want_alloc_mask = __ballot(1);
	uint want_alloc = 1;
	uint leader_lid = __ffs(want_alloc_mask) - 1;
	uint counter_val;
	if(lid == leader_lid)
		counter_val = atomicAdd(counters_g + icounter, 1);
	counter_val = __shfl((int)counter_val, leader_lid);
	int nwords = nblocks_g / 32;
	// initial position
	uint iword = (wid + counter_val * hash_step_g) & (nwords - 1);
	// the allocated pointer
	void *ptr = 0;
	// iterate until successfully reserved
	for(uint i = 0; i < nwords; i++) {
		// try reserve
		if(want_alloc) {
			uint nwant_alloc = __popc(want_alloc_mask);
			// lane number, but only among active lanes. Note that it's inverted, so
			// that the leader lane gets the last portion
			uint leader_lid = __ffs(want_alloc_mask) - 1;
			uint lid_in_active = 	__popc(want_alloc_mask & ((1 << lid) - 1));
			uint reserved_mask;
			if(lid == leader_lid)
				reserved_mask = ~atomicOr(block_bits_g + iword, ~0);
			reserved_mask = __shfl((int)reserved_mask, leader_lid);
			uint nreserved = __popc(reserved_mask);
			// TODO: invent a better way to map threads wanting allocate to
			// reservation mask
			for(int iwant_alloc = 0; iwant_alloc < min(nwant_alloc, nreserved); 
					iwant_alloc++) {
				uint reserved_bit = __ffs(reserved_mask) - 1;
				reserved_mask &= ~(1 << reserved_bit);
				if(iwant_alloc == lid_in_active) {
					ptr = (char *)blocks_g + iword * 32 + reserved_bit;
					want_alloc = 0;
				}
			}
			if(reserved_mask && lid == leader_lid)
				atomicAnd(block_bits_g + iword, ~reserved_mask);
		}
		// else go to the next word
		want_alloc_mask = __ballot(want_alloc);
		if(!want_alloc_mask)
			break;
		iword = (iword + hash_step_g) & (nwords - 1);
	}  // for(i in nwords)
	if(!ptr)
		printf("cannot allocate memory\n");
	return ptr;
}  // hamalloc2

void ha_init(void) {
	// allocate memory
	uint *block_bits, *counters;
	void *blocks;
	cucheck(cudaMalloc(&blocks, NBLOCKS * BLOCK_SZ));
	cucheck(cudaMalloc((void **)&block_bits, NBLOCKS / 32 * sizeof(uint)));
	cucheck(cudaMalloc((void **)&counters, NCOUNTERS * sizeof(uint)));
	// initialize to zero
	cucheck(cudaMemset(block_bits, 0, NBLOCKS / 32 * sizeof(uint)));
	cucheck(cudaMemset(counters, 1, NCOUNTERS * sizeof(uint)));
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
	for(int iptr = tid; iptr < NMALLOCS * NTHREADS2; iptr += NTHREADS2)
		ptrs[iptr] = hamalloc(16);
}  // malloc_k
// read-and-free pointer kernel
__global__ void free_k(void **ptrs) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for(int iptr = tid; iptr < NMALLOCS * NTHREADS2; iptr += NTHREADS2)
		hafree(ptrs[iptr]);
}  // free_k

void run_test1(void) {
	double t1 = omp_get_wtime();
	for(int itry = 0; itry < NTRIES; itry++) {
		malloc_free_k<<<NTHREADS / BS, BS>>>(0);
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
		malloc_k<<<NTHREADS2 / BS, BS>>>(d_ptrs);
		cucheck(cudaGetLastError());
		//cucheck(cudaStreamSynchronize(0));
		free_k<<<NTHREADS2 / BS, BS>>>(d_ptrs);
		cucheck(cudaGetLastError());
		cucheck(cudaStreamSynchronize(0));
	}
	double t2 = omp_get_wtime();
	// copy back and print some allocated pointers
	void **h_ptrs = (void **)malloc(ptrs_sz);
	cucheck(cudaMemcpy(h_ptrs, d_ptrs, ptrs_sz, cudaMemcpyDeviceToHost));
	cucheck(cudaFree(d_ptrs));
	for(int ip = 0; ip < 256; ip += 7)
		printf("d_ptrs[%d] = %p\n", ip, h_ptrs[ip]);
	free(h_ptrs);
	double nmallocs = (double)NMALLOCS * NTHREADS2 * NTRIES;
	printf("test 2 (first all mallocs, then all frees):\n");
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
