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
#define MAX_NSBS 4096
/** the size of SB set, in words; the number of used SBs can be smaller */
#define SB_SET_SZ (MAX_NSBS / WORD_SZ)

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
	//__threadfence();
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

/** find the largest prime number below this one, and not dividing this one */
uint max_prime_below(uint n);

#endif
