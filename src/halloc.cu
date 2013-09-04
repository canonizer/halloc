/** @file halloc.cu implementation of halloc allocator */
#define HALLOCLIB_COMPILING

#include <assert.h>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grid.h"
#include "halloc.h"
#include "sbset.h"
#include "size-info.h"
#include "slab.h"
#include "utils.h"

// global variables used by multiple .cuh files
#include "globals.cuh"

#include "grid.cuh"
#include "sbset.cuh"
#include "size-info.cuh"
#include "slab.cuh"

#define MOSTLY_TID_HASH 0

/** the number of allocation counters */
#if MOSTLY_TID_HASH
#define NCOUNTERS 8192
#else
#define NCOUNTERS MAX_NSIZES
//#define NCOUNTERS 1
#define COUNTER_FREQ (32 * 1024)
#endif

/** thread frequency for initial hashing */
//#define THREAD_FREQ 11
#define THREAD_FREQ 11
/** allocation counter increment */
#define COUNTER_INC 1

/** allocation counters */
__device__ uint counters_g[NCOUNTERS];

/** gets size id from size in bytes */
__device__ inline uint size_id_from_nbytes(uint nbytes) {
	nbytes = max(nbytes, MIN_BLOCK_SZ);
	uint iunit = (32 - __clz(nbytes)) - 5, unit = 1 << (iunit + 3);
	// update unit if it is too small
	if(nbytes > 3 * unit) {
		iunit += 1;
		unit *= 2;
	}	
	// allocation size is either 2 * unit or 3 * unit
	if(nbytes <= 2 * unit)
		return 2 * iunit;
	else
		return 2 * iunit + 1;
	//return (nbytes - MIN_BLOCK_SZ) / BLOCK_STEP;
}  // size_id_from_nbytes

/** increments counter for specific size, does so in warp-aggregating-friendly
		fashion */
__device__ __forceinline__ uint size_ctr_inc(uint size_id) {
	//return atomicAdd(&counters_g[size_id], 1);
	bool want_inc = true;
	uint mask, old_counter, lid = lane_id(), leader_lid, group_mask, change;
	//uint change = 1;
	//while(mask = __ballot(want_inc)) {
	//	if(want_inc) {
	//mask = __ballot(1);
	while(want_inc) {
		mask = __ballot(want_inc);
		leader_lid = warp_leader(mask);
		uint leader_size_id = size_id;
		leader_size_id = __shfl((int)leader_size_id, leader_lid);
		group_mask = __ballot(size_id == leader_size_id);
		if(lid == leader_lid)
			old_counter = atomicAdd(&counters_g[size_id], __popc(group_mask));
		mask &= ~group_mask;
		want_inc = want_inc && size_id != leader_size_id;
		// }
	}  // while
	old_counter = __shfl((int)old_counter, leader_lid);
	change =  __popc(group_mask & ((1 << lid) - 1));
	uint cv = old_counter + change;
	return cv;
}  // sb_ctr_inc

/** procedure for small allocation */
__device__ __forceinline__ void *hamalloc_small(uint nbytes) {
	// the head; having multiple heads actually doesn't help
	//uint ihead = (blockIdx.x / 32) % NHEADS;
	uint ihead = 0;
	uint size_id = size_id_from_nbytes(nbytes);
	uint head_sb = *(volatile uint *)&head_sbs_g[ihead][size_id];
	size_info_t *size_info = &size_infos_g[size_id];

	// the counter is based on warp id
	//uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	//uint wid = tid / WORD_SZ;
	//uint lid = threadIdx.x % WORD_SZ;
	uint lid = lane_id();
	//uint leader_lid = warp_leader(__ballot(1));
	uint leader_lid;
	// TODO: 1 counter per size, distribute increments
	//uint icounter = wid % NCOUNTERS;
	//uint icounter = size_id;
	//uint icounter = 0;
	uint cv = size_ctr_inc(size_id);
	// uint cv;
	// if(lid == leader_lid)
	//  	cv = atomicAdd(&counters_g[icounter], COUNTER_INC);
	// cv = __shfl((int)cv, leader_lid);
	void *p = 0;
	// consider returning back to cubic hash on cv
	// initial position
#if MOSTLY_TID_HASH
	uint ichunk = tid * THREAD_FREQ + cv * cv * (cv + 1);
#else
  //uint ichunk = (icounter + NCOUNTERS * cv) * WARP_SZ + lid;
  //uint ichunk = (icounter + cv) * WARP_SZ + (lid + (uint)clock()) % WARP_SZ;
	uint ichunk = cv;
  //uint cv2 = cv / COUNTER_FREQ;
  //ichunk = ichunk * THREAD_FREQ + cv2 * cv2 * cv2;
	ichunk = ichunk * THREAD_FREQ;
#endif
	ichunk = ichunk * size_info->nchunks_in_block % size_info->nchunks;
  //ichunk = ichunk * size_info->nchunks_in_block & (size_info->nchunks - 1);
	// main allocation loop
	bool want_alloc = true, need_roomy_sb = false;
	// use two-level loop to avoid warplocks
	//uint ntries = 0;
	do {
		if(want_alloc) {
			// try allocating in head superblock
			//head_sb = head_sbs_g[size_id];
			p = sb_alloc_in(ihead, head_sb, ichunk, size_id, need_roomy_sb);
			want_alloc = !p;
			//assert(!want_alloc || need_roomy_sb);
			while(__any(need_roomy_sb)) {
				uint need_roomy_mask = __ballot(need_roomy_sb);
				if(need_roomy_sb) {
					leader_lid = warp_leader(need_roomy_mask);
					uint leader_size_id = size_id;
					leader_size_id = __shfl((int)leader_size_id, leader_lid);
					// here try to check whether a new SB is really needed, and get the
					// new SB
					uint new_head_sb;
					if(lid == leader_lid) {
						//assert(!forced);
						//if(forced)
						//	printf("forced detaching head slab %d\n", head_sb);
						new_head_sb = new_sb_for_size(size_id, size_info->chunk_id, ihead);
					}
					if(size_id == leader_size_id) {
						new_head_sb = __shfl((int)new_head_sb, leader_lid);
						//if(new_head_sb != head_sb)
						//	ntries = 0;
						head_sb = new_head_sb;
						want_alloc = want_alloc && head_sb != SB_NONE;
						need_roomy_sb = false;
					}
				}
			}  // while(need new head superblock)
			//ntries++;
			//assert(ntries < 256);
		}
	} while(__any(want_alloc));
	//__threadfence();
	return p;
}  // hamalloc_small

/** procedure for large allocation */
__device__ __forceinline__ void *hamalloc_large(uint nbytes) {
	return malloc(nbytes);
}  // hamalloc_large

__device__ void *hamalloc(uint nbytes) {
	if(nbytes <= MAX_BLOCK_SZ)
		return hamalloc_small(nbytes);
	else
		return hamalloc_large(nbytes);
} // hamalloc

/** procedure for small free*/
__device__ __forceinline__ void hafree_small(void *p, uint sb_id) {
	uint *alloc_sizes = sb_alloc_sizes(sb_id);
	// TODO: make the read volatile
	uint chunk_sz = *(volatile uint *)&sbs_g[sb_id].chunk_sz;
	//uint chunk_sz = 16;
	uint ichunk = (uint)((char *)p - (char *)sbs_g[sb_id].ptr) / chunk_sz;
	uint nchunks = sb_get_reset_alloc_size(alloc_sizes, ichunk);
	//assert(nchunks != 0);
	//uint size_id = sbs_g[sb_id].size_id;
	// TODO: ensure that no L1 caching takes place
	//uint size_id = sb_size_id(sb_counters_g[sb_id]);
	uint *block_bits = sb_block_bits(sb_id);
	uint iword = ichunk / WORD_SZ, ibit = ichunk % WORD_SZ;
	atomicAnd(block_bits + iword, ~(((1 << nchunks) - 1) << ibit));
	sb_ctr_dec(sb_id, nchunks);
}  // hafree_small

/** procedure for large free */
__device__ __forceinline__ void hafree_large(void *p) {
	return free(p);
}  // hafree_large

__device__ void hafree(void *p) {
	// ignore zero pointer
	if(!p)
		return;
	// get the cell descriptor and other data
	uint icell;
	uint64 cell = grid_cell(p, &icell);
	//uint size_id = grid_size_id(icell, cell, p);
	uint sb_id = grid_sb_id(icell, cell, p);
	if(sb_id != GRID_SB_NONE)
		hafree_small(p, sb_id);
	else
		hafree_large(p);
}  // hafree

void ha_init(halloc_opts_t opts) {
	// TODO: initialize all devices
	// get total device memory (in bytes) & total number of superblocks
	uint64 dev_memory;
	cudaDeviceProp dev_prop;
	int dev;
	cucheck(cudaGetDevice(&dev));
	cucheck(cudaGetDeviceProperties(&dev_prop, dev));
	dev_memory = dev_prop.totalGlobalMem;
	uint sb_sz = 1 << opts.sb_sz_sh;
	uint nsbs = dev_memory / sb_sz;
	cuset(nsbs_g, uint, nsbs);
	cuset(sb_sz_g, uint, sb_sz);
	cuset(sb_sz_sh_g, uint, opts.sb_sz_sh);

	// limit memory available to 3/4 of device memory
	opts.memory = min((uint64)opts.memory, 3ull * dev_memory / 4ull);

	// split memory between halloc and CUDA allocator
	uint64 halloc_memory = opts.halloc_fraction * opts.memory;
	uint64 cuda_memory = opts.memory - halloc_memory;
	cucheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, cuda_memory));

	// allocate a fixed number of superblocks, copy them to device
	uint nsbs_alloc = (uint)min((uint64)nsbs, (uint64)halloc_memory / sb_sz);
	size_t sbs_sz = MAX_NSBS * sizeof(superblock_t);
	superblock_t *sbs = (superblock_t *)malloc(sbs_sz);
	memset(sbs, 0, MAX_NSBS * sizeof(superblock_t));
	uint *sb_counters = (uint *)malloc(MAX_NSBS * sizeof(uint));
	memset(sbs, 0xff, MAX_NSBS * sizeof(uint));
	char *base_addr = (char *)~0ull;
	for(uint isb = 0; isb < nsbs_alloc; isb++) {
		sb_counters[isb] = sb_counter_val(0, false, SZ_NONE, SZ_NONE);
		sbs[isb].size_id = SZ_NONE;
		//sbs[isb].flags = 0;
		sbs[isb].chunk_sz = 0;
		//sbs[isb].chunk_id = SZ_NONE;
		//sbs[isb].state = SB_FREE;
		//sbs[isb].mutex = 0;
		cucheck(cudaMalloc(&sbs[isb].ptr, sb_sz));
		base_addr = (char *)min((uint64)base_addr, (uint64)sbs[isb].ptr);
	}
	//cuset_arr(sbs_g, (superblock_t (*)[MAX_NSBS])&sbs);
	cuset_arr(sbs_g, (superblock_t (*)[MAX_NSBS])sbs);
	cuset_arr(sb_counters_g, (uint (*)[MAX_NSBS])sb_counters);
	// also mark free superblocks in the set
	sbset_t free_sbs;
	memset(free_sbs, 0, sizeof(free_sbs));
	for(uint isb = 0; isb < nsbs_alloc; isb++) {
		uint iword = isb / WORD_SZ, ibit = isb % WORD_SZ;
		free_sbs[iword] |= 1 << ibit;
	}
	free_sbs[SB_SET_SZ - 1] = nsbs_alloc;
	cuset_arr(free_sbs_g, &free_sbs);
	base_addr = (char *)((uint64)base_addr / sb_sz * sb_sz);
	if((uint64)base_addr < dev_memory)
		base_addr = 0;
	else
		base_addr -= dev_memory;
	cuset(base_addr_g, void *, base_addr);

	// allocate block bits and zero them out
	void *bit_blocks, *alloc_sizes;
	uint nsb_bit_words = sb_sz / (BLOCK_STEP * WORD_SZ),
		nsb_alloc_words = sb_sz / (BLOCK_STEP * 4);
	// TODO: move numbers into constants
	uint nsb_bit_words_sh = opts.sb_sz_sh - (4 + 5);
	cuset(nsb_bit_words_g, uint, nsb_bit_words);
	cuset(nsb_bit_words_sh_g, uint, nsb_bit_words_sh);
	cuset(nsb_alloc_words_g, uint, nsb_alloc_words);
	size_t bit_blocks_sz = nsb_bit_words * nsbs * sizeof(uint), 
		alloc_sizes_sz = nsb_alloc_words * nsbs * sizeof(uint);
	cucheck(cudaMalloc(&bit_blocks, bit_blocks_sz));
	cucheck(cudaMemset(bit_blocks, 0, bit_blocks_sz));
	cuset(block_bits_g, uint *, (uint *)bit_blocks);
	cucheck(cudaMalloc(&alloc_sizes, alloc_sizes_sz));
	cucheck(cudaMemset(alloc_sizes, 0, alloc_sizes_sz));
	cuset(alloc_sizes_g, uint *, (uint *)alloc_sizes);

	// set sizes info
	//uint nsizes = (MAX_BLOCK_SZ - MIN_BLOCK_SZ) / BLOCK_STEP + 1;
	uint nsizes = 2 * NUNITS;
	cuset(nsizes_g, uint, nsizes);
	size_info_t size_infos[MAX_NSIZES];
	memset(size_infos, 0, MAX_NSIZES * sizeof(size_info_t));
	for(uint isize = 0; isize < nsizes; isize++) {
		uint iunit = isize / 2, unit = 1 << (iunit + 3);
		size_info_t *size_info = &size_infos[isize];
		//size_info->block_sz = isize % 2 ? 3 * unit : 2 * unit;
		uint block_sz = isize % 2 ? 3 * unit : 2 * unit;
		uint nblocks = sb_sz / block_sz;
		size_info->chunk_id = isize % 2 + (isize < nsizes / 2 ? 0 : 2);
		size_info->chunk_sz = (size_info->chunk_id % 2 ? 3 : 2) * 
			(size_info->chunk_id / 2 ? 128 : 8);
		size_info->nchunks_in_block = block_sz / size_info->chunk_sz;
		size_info->nchunks = nblocks * size_info->nchunks_in_block;
		// TODO: use a better hash step
		size_info->hash_step = size_info->nchunks_in_block *
		 	max_prime_below(nblocks / 256 + nblocks / 64);
		//size_info->hash_step = size_info->nchunks_in_block * 17;
		//printf("block = %d, step = %d, nchunks = %d\n", 
		//			 block_sz, size_info->hash_step, size_info->nchunks);
		size_info->roomy_threshold = opts.roomy_fraction * size_info->nchunks;
		size_info->busy_threshold = opts.busy_fraction * size_info->nchunks;
		size_info->sparse_threshold = opts.sparse_fraction * size_info->nchunks;
	}  // for(each size)
	cuset_arr(size_infos_g, &size_infos);

	// set grid info
	uint64 sb_grid[2 * MAX_NSBS];
	for(uint icell = 0; icell < 2 * MAX_NSBS; icell++) 
		sb_grid[icell] = grid_cell_init();
	for(uint isb = 0; isb < nsbs_alloc; isb++)
		grid_add_sb(sb_grid, base_addr, isb, sbs[isb].ptr, sb_sz);
	cuset_arr(sb_grid_g, &sb_grid);
	
	// zero out sets (but have some of the free set)
	//fprintf(stderr, "started cuda-memsetting\n");
	//cuvar_memset(unallocated_sbs_g, 0, sizeof(unallocated_sbs_g));
	cuvar_memset(roomy_sbs_g, 0, sizeof(roomy_sbs_g));
	cuvar_memset(sparse_sbs_g, 0, sizeof(sparse_sbs_g));
	//cuvar_memset(roomy_sbs_g, 0, (MAX_NSIZES * SB_SET_SZ * sizeof(uint)));
	cuvar_memset(head_sbs_g, ~0, sizeof(head_sbs_g));
	cuvar_memset(cached_sbs_g, ~0, sizeof(head_sbs_g));
	cuvar_memset(head_locks_g, 0, sizeof(head_locks_g));
	//cuvar_memset(counters_g, 1, sizeof(counters_g));
	cuvar_memset(counters_g, 11, sizeof(counters_g));
	//fprintf(stderr, "finished cuda-memsetting\n");
	cucheck(cudaStreamSynchronize(0));

	// free all temporary data structures
	free(sbs);
	free(sb_counters);

}  // ha_init

void ha_shutdown(void) {
	// TODO: free memory
}  // ha_shutdown
