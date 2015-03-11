#ifndef HALLOC_SIZE_INFO_H_
#define HALLOC_SIZE_INFO_H_

/** @file size-info.h information and definitions related to sizes */

#include "utils.h"

/** size information type; this is non-changing information, to be stored in
		constant memory */
typedef struct {
	/** number of chunks in slab */
	uint nchunks;
	/** size of a single chunk */
	uint chunk_sz;
	/** id of the chunk to which the size belongs */
	uint chunk_id;
	/** number of chunks in a block for this size */
	uint nchunks_in_block;
	/** threshold (in chunks) for the slab to be declared "sparse", so that it can
		be reused by other sizes with the same chunk size */
	uint sparse_threshold;
	/** step for the hash function */
	uint hash_step;
	/** threshold (in chunks) for the slab to be declared "roomy" */
	uint roomy_threshold;
	/** threshold (in chunks) for the slab to be declared "busy" and be detached */
	uint busy_threshold;
} size_info_t __attribute__((aligned(32)));

/** maximum number of sizes supported */
#define MAX_NSIZES 64
/** maximum number of different chunk sizes supported */
#define MAX_NCHUNK_IDS 8
/** a "no-size" constant */
#define SZ_NONE (~0)
/** block step (16 bytes by default), a power of two */
#define BLOCK_STEP 16
/** minimum unit size (allocation blocks are either 2 or 3 units) */
#define MIN_UNIT_SZ 8
/** maximum unit size */
#define MAX_UNIT_SZ 1024
/** unit step */
#define UNIT_STEP 2
/** the number of units */
#define NUNITS 8
/** minimum block size */
#define MIN_BLOCK_SZ 16
/** maximum block size */
#define MAX_BLOCK_SZ 3072

// chunk manipulation
uint chunk_val(uint chunk_sz) {
	//return chunk_sz;
	uint div3 = chunk_sz % 3 ? 1 : 3;
	if(chunk_sz % 3 == 0)
		chunk_sz /= 3;
	uint sh = 0;
	for(; (1 << sh) < chunk_sz; sh++);
	return div3 << 16 | sh;
}

__host__ __device__ inline uint chunk_mul(uint v, uint chunk_sz) {
	//return v * chunk_sz;
	return (v << (chunk_sz & 0xffffu)) * (chunk_sz >> 16);
}

__host__ __device__ inline uint chunk_div(uint v, uint chunk_sz) {
	// return v / chunk_sz
	if(chunk_sz >> 16u == 3u)
		v /= 3u;
	return v >> (chunk_sz & 0xffffu);
}

/** loads size-related data with appropriate caching */
__device__ __forceinline__ uint ldsz(const uint *p) {
#if __CUDA_CC__ >= 500 
	// on Maxwell, p points to constant memory, just use it
	return *p;
#else
	// below Maxwell, use L1 caching if available
	return ldca(p);
#endif
}  // ldsz

/** gets the size infos as they are stored */
__device__ __forceinline__ const size_info_t *info_for_size(uint size_id);

/** easy way to get a size field */
#define sz_get(szinfo, field) \
	(ldsz(&szinfo->field))
#define szid_get(size_id, field) \
	sz_get(info_for_size(size_id), field)

#endif
