#ifndef HALLOC_UTILS_H_
#define HALLOC_UTILS_H_

/** @file utils.h some utility macros, functions and definitions */

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

/** sets CUDA device variable */
#define cuset(symbol, T, val)																		\
{																																\
	void *cuset_addr;																							\
	cucheck(cudaGetSymbolAddress(&cuset_addr, symbol));						\
	T cuset_val = (val);																					\
	cucheck(cudaMemcpy(cuset_addr, &cuset_val, sizeof(cuset_val), \
										 cudaMemcpyHostToDevice));									\
}  // cuset

/** gets the value of the CUDA device variable */
#define cuget(pval, symbol)																\
{																													\
	void *cuget_addr;																				\
	cucheck(cudaGetSymbolAddress(&cuget_addr, symbol));			\
	cucheck(cudaMemcpy((pval), cuget_addr, sizeof(*(pval)), \
										 cudaMemcpyDeviceToHost));						\
}

#define cuset_arr(symbol, val)												\
{																											\
	void *cuset_addr;																		\
	cucheck(cudaGetSymbolAddress(&cuset_addr, symbol));	\
	cucheck(cudaMemcpy(cuset_addr, *val, sizeof(*val),	\
										 cudaMemcpyHostToDevice));				\
} // cuset_arr

/** acts as cudaMemset(), but accepts device variable */
#define cuvar_memset(symbol, val, sz)									\
{																											\
	void *cuvar_addr;																		\
	cucheck(cudaGetSymbolAddress(&cuvar_addr, symbol));	\
	cucheck(cudaMemset(cuvar_addr, val, sz));						\
}  // cuvar_memset

/** 64-bit integer type */
typedef unsigned long long uint64;

// constants
/** word size (the word is uint, which is assumed to be 32-bit) */
#define WORD_SZ 32
/** the warp size (32 on current NVidia architectures) */
#define WARP_SZ 32
/** maximum number of superblocks */
//#define MAX_NSBS 4096
#define MAX_NSBS 8192
/** the size of SB set, in words; the number of used SBs can be smaller */
#define SB_SET_SZ (MAX_NSBS / WORD_SZ)
/** the maximum number of warps in a thread block */
#define MAX_NWARPS 32

/** division with rounding upwards, useful for kernel calls */
inline __host__ __device__ int divup
(int a, int b) { return a / b + (a % b ? 1 : 0); }

/** checks whether the step is in mask */
__device__ inline bool step_is_in_mask(uint mask, uint val) {
	return (mask >> val) & 1;
}

/** gets the distance to the next higher mask value	*/
__device__ inline uint step_next_dist(uint mask, uint val) {
	uint res =  __ffs(mask >> (val + 1));
	return res ? res : WORD_SZ - val;
}

/** tries single-thread-per-warp lock 
		@returns true if locking is successful and false otherwise
 */
__device__ inline bool try_lock(uint *mutex) {
	return atomicExch(mutex, 1) == 0;
}
/** single-thread-per-warp lock; loops until the lock is acquired */
__device__ inline void lock(uint *mutex) {
	while(!try_lock(mutex));
}
/** single-thread-per-warp unlock, without threadfence */
__device__ inline void unlock(uint *mutex) {
	__threadfence();
	atomicExch(mutex, 0);
}
/** waits until the mutex is unlocked, but does not attempt locking */
__device__ inline void wait_unlock(uint *mutex) {
	while(*(volatile uint *)mutex);
	// {
	// 	uint64 t1 = clock64();
	// 	while(clock64() - t1 < 1);
	// }
}
/** gets the warp leader based on the mask */
__device__ inline uint warp_leader(uint mask) {
	return __ffs(mask) - 1;
}

/** gets the lane id inside the warp */
__device__ inline uint lane_id(void) {
	uint lid;
	asm("mov.u32 %0, %%laneid;" : "=r" (lid));
	return lid;
	// TODO: maybe use more reliable lane id computation
	//return threadIdx.x % WARP_SZ;
}

/** gets the id of the warp */
__device__ inline uint warp_id(void) {
	// TODO: use something more stable
	return threadIdx.x / WARP_SZ;
}

/** broadcasts a value to all participating threads in a warp */
__device__ inline uint warp_bcast(uint v, uint root_lid) {
#if __CUDA_ARCH__ >= 300
	// use warp intrinsics
	return (uint) __shfl((int)v, root_lid);
#else
	// use shared memory
	volatile __shared__ uint vs[MAX_NWARPS];
	if(lane_id() == root_lid)
		vs[warp_id()] = v;
	return vs[warp_id()];
#endif
}  // warp_bcast

/** loads the data with caching */
__device__ inline uint ldca(const uint *p) {
	uint res;
	asm("ld.global.ca.u32 %0, [%1];": "=r"(res) : "l"(p));
	return res;
}  

__device__ inline uint64 ldca(const uint64 *p) {
	uint64 res;
	asm("ld.global.ca.u64 %0, [%1];": "=l"(res) : "l"(p));
	return res;
}  

__device__ inline void *ldca(void * const *p) {
	void *res;
	asm("ld.global.ca.u64 %0, [%1];": "=l"(res) : "l"(p));
	return res;
}  

/** prefetches into L1 cache */
__device__ inline void prefetch_l1(const void *p) {
	asm("prefetch.global.L1 [%0];": :"l"(p));
}

/** prefetches into L2 cache */
__device__ inline void prefetch_l2(const void *p) {
	asm("prefetch.global.L2 [%0];": :"l"(p));
}

__device__ inline uint lanemask_lt() {
	uint mask;
	asm("mov.u32 %0, %%lanemask_lt;" : "=r" (mask));
	return mask;
}

/** find the largest prime number below this one, and not dividing this one */
uint max_prime_below(uint n, uint nb);

#endif
