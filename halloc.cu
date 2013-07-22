/** @file hamalloc.cu implementation of halloc allocator */

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

#include "grid.cuh"
#include "sbset.cuh"
#include "size-info.cuh"

#include "slab.cuh"

/** the number of allocation counters*/
#define NCOUNTERS 4096
/** thread frequency for initial hashing */
#define THREAD_FREQ 11
/** allocation counter increment */
#define COUNTER_INC 1
/** maximum amount of memory to allocate, in MB */
#define MAX_ALLOC_MEM 512

/** allocation counters */
__device__ uint counters_g[NCOUNTERS];

/** gets size id from size in bytes */
__device__ inline uint size_id_from_nbytes(uint nbytes) {
	nbytes = max(nbytes, MIN_BLOCK_SZ);
	uint iunit = __ffs(nbytes) - 5, unit = 1 << (iunit + 3);
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

__device__ void *hamalloc(uint nbytes) {
	if(nbytes > MAX_BLOCK_SZ)
		return 0;
	// ignore zero-sized allocations
	uint size_id = size_id_from_nbytes(nbytes);
	uint head_sb = head_sbs_g[size_id];
	size_info_t size_info = size_infos_g[size_id];
	// the counter is based on block id
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint wid = tid / WORD_SZ, lid = tid % WORD_SZ;
	uint leader_lid = warp_leader(__ballot(1));
	uint icounter = wid % NCOUNTERS;
	uint cv;
	if(lid == leader_lid)
		cv = atomicAdd(&counters_g[icounter], COUNTER_INC);
	cv = __shfl((int)cv, leader_lid);
	void *p = 0;
	// initial position
	// TODO: use a real but cheap random number generator
	// uint iblock = (tid * THREAD_FREQ + cv1 + cv2 * cv2 * (cv2 + 1)) %
	//  	size_infos_g[size_id].nblocks;
	// (icounter * WARP_SZ + lid) also provides good initial value qualities
	// using xor instead of multiplication can provide even higher entropy
	//uint cv2 = cv >> 3, cv1 = cv & 7;
	uint iblock = (tid * THREAD_FREQ + 
							 ((cv * cv) * (cv + 1))) % size_info.nblocks;
	//uint iblock = (tid * THREAD_FREQ + cv * cv * (cv + 1)) & (size_info.nblocks - 1);
	// main allocation loop
	bool want_alloc = true;
	// use two-level loop to avoid warplocks
	do {
		if(want_alloc) {
			// try allocating in head superblock
			//head_sb = head_sbs_g[size_id];
			p = sb_alloc_in(head_sb, iblock, size_info, size_id);
			bool need_roomy_sb = want_alloc = !p;
			while(__any(need_roomy_sb)) {
				uint need_roomy_mask = __ballot(need_roomy_sb);
				if(need_roomy_sb) {
					leader_lid = warp_leader(need_roomy_mask);
					uint leader_size_id = __shfl((int)size_id, leader_lid);
					// here try to check whether a new SB is really needed, and get the
					// new SB
					if(lid == leader_lid)
						head_sb = new_sb_for_size(size_id);
					if(size_id == leader_size_id) {
						head_sb = __shfl((int)head_sb, leader_lid);
						want_alloc = head_sb != SB_NONE;
						need_roomy_sb = false;
					}
				}
			}  // while(need new head superblock)
			// just quickly return 0
			// return 0;
		}
	} while(__any(want_alloc));
	//if(!p)
	//	printf("cannot allocate memory\n");
	return p;
}  // hamalloc

__device__ void hafree(void *p) {
	// ignore zero pointer
	if(!p)
		return;
	// get the cell descriptor and other data
	uint icell;
	uint64 cell = grid_cell(p, &icell);
	//uint size_id = grid_size_id(icell, cell, p);
	uint sb_id = grid_sb_id(icell, cell, p);
	//uint *alloc_sizes = sb_alloc_sizes(sb_id);
	//uint ichunk = (uint)((char *)p - (char *)sbs_g[sb_id].ptr) / BLOCK_STEP;
	//uint size_id = sb_get_reset_alloc_size(alloc_sizes, ichunk);
	uint size_id = sbs_g[sb_id].size_id;
	uint *block_bits = sb_block_bits(sb_id);
	// free the memory
	// TODO: this division is what eats all performance
	// replace it with reciprocal multiplication
	uint iblock = (uint)((char *)p - (char *)sbs_g[sb_id].ptr) /
			size_infos_g[size_id].block_sz;
	uint iword = iblock / WORD_SZ, ibit = iblock % WORD_SZ;
	//uint new_word = atomicAnd(block_bits + iword, ~(1 << ibit)) & ~(1 << ibit);
	atomicAnd(block_bits + iword, ~(1 << ibit));
	//printf("freeing: sb_id = %d, p = %p, iblock = %d\n", sb_id, p, iblock);
	//sb_dctr_dec(size_id, sb_id, new_word, iword);
	//sb_ctr_dec(size_id, sb_id, size_infos_g[size_id].block_sz / BLOCK_STEP);
	sb_ctr_dec(size_id, sb_id, 1);
}  // hafree

void ha_init(void) {
	// TODO: initialize all devices
	// get total device memory (in bytes) & total number of superblocks
	uint64 dev_memory;
	cudaDeviceProp dev_prop;
	int dev;
	cucheck(cudaGetDevice(&dev));
	cucheck(cudaGetDeviceProperties(&dev_prop, dev));
	dev_memory = dev_prop.totalGlobalMem;
	uint nsbs = dev_memory / SB_SZ, sb_sz = SB_SZ;
	cuset(nsbs_g, uint, nsbs);
	cuset(sb_sz_g, uint, sb_sz);
	cuset(sb_sz_sh_g, uint, SB_SZ_SH);

	// allocate a fixed number of superblocks, copy them to device
	uint nsbs_alloc = (uint)min((uint64)nsbs, 
												(uint64)MAX_ALLOC_MEM * 1024 * 1024 / sb_sz);
	size_t sbs_sz = MAX_NSBS * sizeof(superblock_t);
	superblock_t *sbs = (superblock_t *)malloc(sbs_sz);
	memset(sbs, 0, MAX_NSBS * sizeof(superblock_t));
	uint *sb_counters = (uint *)malloc(MAX_NSBS * sizeof(uint));
	memset(sbs, 0xff, MAX_NSBS * sizeof(uint));
	char *base_addr = (char *)~0ull;
	for(uint isb = 0; isb < nsbs_alloc; isb++) {
		sb_counters[isb] = sb_counter_val(0, false, SZ_NONE, SZ_NONE);
		sbs[isb].size_id = SZ_NONE;
		//sbs[isb].chunk_id = SZ_NONE;
		//sbs[isb].state = SB_FREE;
		//sbs[isb].mutex = 0;
		cucheck(cudaMalloc(&sbs[isb].ptr, SB_SZ));
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
	cuset_arr(free_sbs_g, &free_sbs);
	base_addr = (char *)((uint64)base_addr / SB_SZ * SB_SZ);
	if((uint64)base_addr < dev_memory)
		base_addr = 0;
	else
		base_addr -= dev_memory;
	cuset(base_addr_g, void *, base_addr);

	// allocate block bits and zero them out
	void *bit_blocks, *alloc_sizes;
	uint nsb_bit_words = SB_SZ / (BLOCK_STEP * WORD_SZ), 
		nsb_alloc_words = SB_SZ / (BLOCK_STEP * 4);
	// TODO: make move numbers into constants
	uint nsb_bit_words_sh = SB_SZ_SH - (4 + 5);
	cuset(nsb_bit_words_g, uint, nsb_bit_words);
	cuset(nsb_bit_words_sh_g, uint, nsb_bit_words_sh);
	cuset(nsb_alloc_words_g, uint, nsb_alloc_words);
	size_t bit_blocks_sz = nsb_bit_words * nsbs * sizeof(uint), 
		alloc_sizes_sz = nsb_alloc_words * nsbs * sizeof(uint);
	cucheck(cudaMalloc(&bit_blocks, bit_blocks_sz));
	cucheck(cudaMemset(bit_blocks, 0, bit_blocks_sz));
	cuset(block_bits_g, uint *, (uint *)bit_blocks);
	cucheck(cudaMalloc(&alloc_sizes, alloc_sizes_sz));
	cucheck(cudaMemset(alloc_sizes, 0xff, alloc_sizes_sz));
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
		size_info->block_sz = isize % 2 ? 3 * unit : 2 * unit;
		size_info->nblocks = sb_sz / size_info->block_sz;
		size_info->hash_step = 
		 	max_prime_below(size_info->nblocks / 256 + size_info->nblocks / 64);
		//size_info->hash_step = size_info->nblocks / 256 + size_info->nblocks / 64 + 1;
		size_info->roomy_threshold = 0.25 * size_info->nblocks;
		size_info->busy_threshold = 0.8 * size_info->nblocks;
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
	//cuvar_memset(roomy_sbs_g, 0, (MAX_NSIZES * SB_SET_SZ * sizeof(uint)));
	cuvar_memset(head_sbs_g, ~0, sizeof(head_sbs_g));
	cuvar_memset(head_locks_g, 0, sizeof(head_locks_g));
	cuvar_memset(counters_g, 1, sizeof(counters_g));
	//fprintf(stderr, "finished cuda-memsetting\n");
	cucheck(cudaStreamSynchronize(0));

	// free all temporary data structures
	free(sbs);
	free(sb_counters);

}  // ha_init

void ha_shutdown(void) {
	// TODO: free memory
}  // ha_shutdown
