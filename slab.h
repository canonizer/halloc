#ifndef HALLOC_SLAB_H_
#define HALLOC_SLAB_H_

/** @file slab.h slab (superblock) header file */
/** superblock descriptor type; information is mostly changing; note that during
		allocation, a superblock is mostly identified by superblock id */
typedef struct {
	/** superblock size id */
	uint size_id;
	/** pointer to memory owned by superblock */
	void *ptr;
} superblock_t;

/** a step to check whether the slab can be moved to another free category */
#define SB_FREE_STEP 2048
/** maximum number of tries inside a superblock after which the allocation
		attempt is abandoned */
#define MAX_NTRIES 256
/** a "no-sb" constant */
#define SB_NONE (~0)
/** default superblock size, in bytes */
#define SB_SZ_SH (10 + 10 + 2)
#define SB_SZ (1 << SB_SZ_SH)

/** positions and sizes related to slab counters */
#define SB_SIZE_POS 0
#define SB_SIZE_SZ 6
#define SB_CHUNK_POS 6
#define SB_CHUNK_SZ 3
#define SB_COUNT_POS 10
#define SB_COUNT_SZ 22


// functions for manipulation with counter values
/** gets slab allocation count */
__device__ inline uint sb_count(uint counter) {
	return counter >> SB_COUNT_POS;
}
/** gets size id  */
__device__ inline uint sb_size_id(uint counter) {
	return (counter >> SB_SIZE_POS) & ((1 << SB_SIZE_SZ) - 1);
}
/** gets chunk size id */
__device__ inline uint sb_chunk_id(uint counter) {
	return (counter >> SB_CHUNK_POS) & ((1 << SB_CHUNK_SZ) - 1);
}
/** gets the counter value for the specified count, size id and chunk id */
__host__ __device__ inline uint sb_counter_val
(uint count, uint chunk_id, uint size_id) {
	return count << SB_COUNT_POS | 
		chunk_id & ((1 << SB_CHUNK_SZ) - 1) << SB_CHUNK_POS | 
		size_id & ((1 << SB_SIZE_SZ) - 1) << SB_SIZE_POS;
}
/** atomically increments/decrements slab counter, returns old slab counter value */
__device__ inline uint sb_counter_inc(uint *counter, uint change) {
	return atomicAdd(counter, change << SB_COUNT_POS);
}
__device__ inline uint sb_counter_dec(uint *counter, uint change) {
	return atomicSub(counter, change << SB_COUNT_POS);
}

#endif
