/** @file hamalloc.cu implementation of halloc allocator */

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "halloc.h"

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

/** 64-bit integer type */
typedef unsigned long long uint64;

// constants

/** default hash step */
#define HASH_STEP (NBLOCKS / 256 + NBLOCKS / 64 - 1)
/** the number of allocation counters*/
#define NCOUNTERS 2048
/** thread frequency for initial hashing */
#define THREAD_FREQ 17
/** allocation counter increment */
#define COUNTER_INC 1
/** word size (the word is uint, which is assumed to be 32-bit) */
#define WORD_SZ 32
/** the warp size (32 on current NVidia architectures) */
#define WARP_SZ 32
/** maximum number of superblocks */
#define MAX_NSBS 2048
/** number of superblocks to allocate initially */
#define NSBS_ALLOC 64
/** the size of SB set, in words; the number of used SBs can be smaller */
#define SB_SET_SZ (MAX_NSBS / WORD_SZ)
/** maximum number of sizes supported */
#define MAX_NSIZES 64
/** whether to do head check after allocation */
#define HEAD_SB_CHECK 0
/** number of superblock counters */
#define NSB_COUNTERS 16
/** a step to change the value of a distributed counter */
#define SB_DISTR_STEP 8
/** a step to change the value of main counter */
#define SB_MAIN_STEP 512
/** maximum number of tries inside a superblock after which the allocation
		attempt is abandoned */
#define MAX_NTRIES 64
/** a "no-sb" constant */
#define SB_NONE (~0)
/** block step (16 bytes by default), a power of two */
#define BLOCK_STEP 16
/** minimum block size (a power of two) */
#define MIN_BLOCK_SZ 16
/** maximum block size (a power of two) */
#define MAX_BLOCK_SZ 256
/** default superblock size, in bytes */
#define SB_SZ (4 * 1024 * 1024)

// types 

/** size information type; this is non-changing information, to be stored in
		constant memory */
typedef struct {
/** block size */
uint block_sz;
/** number of blocks in superblock */
uint nblocks_in_sb;
/** step for the hash function */
uint hash_step;
/** threshold for the superblock to be declared "roomy" */
uint roomy_threshold;
/** threshold for the superblock to be declared "busy" and become candidate for
		detachment */
uint busy_threshold;
} size_info_t;

/** superblock descriptor type; information is mostly changing; note that during
		allocation, a superblock is mostly identified by superblock id */
typedef struct {
	/** counter of occupied blocks */
	uint noccupied;
	/** superblock id, just in case */
	uint id;
	/** pointer to memory owned by superblock */
	void *ptr;
} superblock_t;

/** superblock set type */
typedef uint sbset_t[SB_SET_SZ];

// constant global variables (data pointed to by pointers can still be mutated)
/** real possible number of superblocks (based on device memory and superblock
		size) */
__constant__ uint nsbs_g;
/** real number of sizes */
__constant__ uint nsizes_g;
/** block bits for all superblocks (content may change) */
uint * __constant__ block_bits_g;
/** number of block bit words per superblock */
__constant__ uint nsb_bit_words_g;
/** information on sizes */
__constant__ size_info_t size_infos_g[MAX_NSIZES];
/** base address of the grid; this is the start address of the grid. It is
		always aligned to superblock size boundary */
__constant__ void *base_addr_g;
/** superblock size (common for all superblocks, power-of-two) */
__constant__ uint sb_sz_g;

// mutable global variables
/** the set of all unallocated superblocks */
__device__ sbset_t unallocated_sbs_g;
/** the set of all free superblocks */
__device__ sbset_t free_sbs_g;
/** the set of "roomy" superblocks for the size */
__device__ sbset_t roomy_sbs_g[MAX_NSIZES];
/** head superblocks for each size */
__device__ uint head_sbs_g[MAX_NSIZES];
/** superblock operation locks per size */
__device__ uint head_locks_g[MAX_NSIZES];
/** superblock descriptors */
__device__ superblock_t sbs_g[MAX_NSBS];
/** superblock distributed bookkeeping counters */
__device__ uint sb_counters_g[MAX_NSBS][NSB_COUNTERS];
/** allocation counters */
__device__ uint counters_g[NCOUNTERS];
/** superblock grid */
__device__ uint64 sb_grid_g[2 * MAX_NSBS];

/** gets superblock from set (and removes it) */
__device__ inline uint sbset_get_from(sbset_t *sbset, uint except) {
	// TODO: maybe do several trials to be sure
	for(uint iword = 0; iword < nsbs / WORD_SZ; iword++)
		for(uint iword = sbset[iword], ibit = __ffs(iword) - 1; iword; 
				iword &= ~(1 <<	ibit), ibit = __ffs(iword) - 1) {
			// try locking the bit
			if(iword * WORD_SZ + ibit != except) {
				uint mask = 1 << ibit;
				if(atomicAnd(&sbset[iword], ~mask) & mask)
					return iword * WORD_SZ + ibit;
			}
		}
	return SB_NONE;
}  // sbset_get_from

/** adds ("returns") superblock to the set */
__device__ inline uint sbset_add_to(sbset_t *sbset, uint sb) {
	uint iword = sb / WORD_SZ, ibit = sb % WORD_SZ;
	atomicOr(&sbset[iword], 1 << ibit);
}  // sbset_add_to

// constants related to grid cells
#define GRID_SIZE_LEN 6
#define GRID_ADDR_LEN 20
#define GRID_SB_LEN 13
#define GRID_INIT_POS 0
#define GRID_FIRST_SIZE_POS 1
#define GRID_SECOND_SIZE_POS 7
#define GRID_FIRST_SB_POS 13
#define GRID_SECOND_SB_POS 26
#define GRID_ADDR_POS 39
#define GRID_ADDR_UNIT_SH 4
/** checks whether the grid cell is initialized */
__device__ inline bool grid_is_init(uint64 cell) {
	return (cell >> GRID_INIT_POS) & 1;
} 
/** gets the first size id of the grid cell */
__device__ inline uint grid_first_size_id(uint64 cell) {
	return (cell >> GRID_FIRST_SIZE_POS) & ((1 << GRID_SIZE_LEN) - 1);
}
/** gets the  second size id of the grid cell */
__device__ inline uint grid_second_size_id(uint64 cell) {
	return (cell >> GRID_SECOND_SIZE_POS) & ((1 << GRID_SIZE_LEN) - 1);
}
/** gets the first superblock id of the grid cell  */
__device__ inline uint grid_first_sb_id(uint64 cell) {
	return (cell >> GRID_FIRST_SB_POS) & ((1 << GRID_SB_LEN) - 1);
}
/** gets the second superblock id of the grid cell  */
__device__ inline uint grid_second_sb_id(uint64 cell) {
	return (cell >> GRID_SECOND_SB_POS) & ((1 << GRID_SB_LEN) - 1);
}
/** gets the mid-address of the grid cell */
__device__ inline void *grid_mid_addr(uint icell, uint64 cell) {
	uint in_sb_addr = ((cell >> GRID_ADDR_POS) & ((1 << GRID_ADDR_LEN) - 1)) 
		<< GRID_ADDR_UNIT_SH;
	return (char *)base_addr_g + icell * sb_sz_g;
}
/** gets the grid cell for the pointer */
__device__ inline uint64 grid_cell(void *p, uint *icell) {
	// TODO: handle stale cell data
	// TODO: optimize for power-of-two SB sizes
	*icell = ((char *)p - (char *)base_addr_g) / sb_sz_g;
	return sb_grid_g[*icell];
}
/** gets the (de)allocation size id for the pointer */
__device__ inline uint grid_size_id(uint icell, uint64 cell, void *p) {
	void *midp = grid_mid_addr(icell, cell);
	return p < midp ? grid_first_size_id(cell) : grid_second_size_id(cell);
}
/** gets the (de)allocation superblock id for the pointer */
__device__ inline uint grid_sb_id(uint icell, uint64 cell, void *p) {
	void *midp = grid_mid_addr(icell, cell);
	return p < midp ? grid_first_sb_id(cell) : grid_second_sb_id(cell);
}

/** gets block bits for superblock */
__device__ inline uint *sb_block_bits(uint sb) {
	return bit_blocks_g + sb * nsb_bit_words_g;
}  // sb_block_bits

/** increment distributed superblock counter 
		@param size_id the size id for allocation; ignored 
		@param sb the superblock id
		@param old_word the old value of the word where allocation took place
		@param iword the index of the word where allocation took place
*/
void sb_dctr_inc(uint size_id, uint sb, uint old_word, uint iword) {
	uint nword_blocks = __popc(old_word);
	if(nword_blocks % SB_DISTR_STEP == 0) {
		// increment distributed counter
		uint ictr = iword % NSB_COUNTERS;
		uint old_val = atomicAdd(&sb_counters_g[sb][ictr], SB_DISTR_STEP);
		if(old_val % SB_MAIN_STEP == 0) {
			// increment main superblock counter; the final value is not needed
			atomicAdd(&sbs_g[sb].noccupied, SB_MAIN_STEP);
		}
	}
}  // sb_dctr_inc

/** decrement distributed superblock counter 
		@param size_id the size id for the deallocation
		@param sb the superblock id
		@param new_word the new value of the word where the allocation took place
		@param iword the index of the word where the allocation took place
 */
void inline sb_dctr_dec(uint size_id, uint sb, uint new_word, uint iword) {
	uint nword_blocks = __popc(new_word);
	if(nword_blocks % SB_DISTR_STEP == 0) {
		// decrement distributed counter
		uint ictr = iword % NSB_COUNTERS;
		uint new_val = atomicAdd(&sb_counters_g[sb][ictr], -SB_DISTR_STEP) - 
			SB_DISTR_STEP;
		if(new_val % SB_MAIN_STEP == 0) {
			// decrement main counter
			new_main_val = atomicAdd(&sbs_g[sb].noccupied, -SB_MAIN_STEP) - 
				SB_MAIN_STEP;
			if(new_main_val <= size_infos_g[size_id].roomy_threshold) {
				// mark superblock as roomy for current size
				sbset_add_to(&roomy_sbs_g[size_id], sb);
			}
		}
	}
}  // sb_dctr_dec

/** allocates memory inside the superblock 
		@param sb the superblock inside which to allocate
		@param [in,out] iblock the block from which to start searching
		@param size_id the size id for the allocation
		@returns the pointer to the allocated memory, or 0 if unable to allocate
*/
__device__ inline void *sb_alloc_in(uint sb, uint *iblock, uint size_id) {
	if(sb == SB_NONE)
		return 0;
	void *p = 0;
	uint *block_bits = sb_block_bits(sb);
	// check the superblock occupancy counter
	uint noccupied = sbs_g[sb].noccupied;
	if(noccupied >= size_infos_g[size_id].busy_threshold)
		return 0;
	uint old_word, iword;
	// iterate until successfully reserved
	for(uint itry = 0; itry < MAX_NTRIES; itry++) {
		//for(uint i = 0; i < 1; i++) {
		// try reserve
		iword = *iblock / WORD_SZ;
		uint ibit = *iblock % WORD_SZ;
		uint alloc_mask = 1 << ibit;
		old_word = atomicOr(block_bits + iword, alloc_mask);
		if(!(old_word & alloc_mask)) {
			// reservation successful, return pointer
			ptr = (char *)sbs_g[sb].ptr + *iblock * size_infos_g[size_id].block_sz;
			break;
		} else 
			*iblock = (*iblock + size_infos_g[size_id].hash_step) 
				& (size_infos_g[size_id].nblocks - 1);
	}
	if(ptr)
		sb_dctr_inc(size_id, sb, old_word, iword);
	return ptr;	
}  // sb_alloc_in

/** tries to find a new superblock for the given size
		@returns the new head superblock id if found, and SB_NONE if none
		@remarks this function should be called by at most 1 thread in a warp at a time
*/
__device__ inline uint new_sb_for_size(uint size_id) {
	// try locking size id
	if(!atomicExch(&head_locks_g[size_id], 1)) {
		// locked successfully, check if really need replacing blocks
		uint cur_head = *(volatile uint *)&head_sbs_g[size_id];
		uint new_head;
		if(sb_id == SB_NONE || 
			 *(volatile uint *)&sbs_g[cur_head].noccupied >=
			 size_infos_g[size_id].busy_threshold) {
			// replacement really necessary; first try among roomy sb's of current 
			// size
			uint new_head = sbset_get_from(&roomy_sbs_g[size_id], cur_head);
			if(new_head == SB_NONE) {
				// try getting from free superblocks
				new_head = sbset_get_from(&free_sbs_g, SB_NONE);
			}
			if(new_head != SB_NONE) {
				// replace current head
				head_sbs_g[size_id] = new_head;
			}
		} else {
			// just re-read the new head superblock
			new_head = *(volatile uint *)&head_sbs_g[size_id];
		}
		// unlock
		__threadfence();
		atomicExch(&head_locks_g[size_id], 0);
		return new_head;
	} else {
		// someone else working on current head superblock; 
		while(*(volatile uint *)&head_locks_g[size_id]);
		return *(volatile uint)&head_sbs_g[size_id];
	}
}  // new_sb_for_size

__device__ void *hamalloc(size_t nbytes) {
	// ignore zero-sized allocations
	if(!nbytes)
		return 0;
	uint size_id = (nbytes - MIN_BLOCK_SZ) / BLOCK_STEP;
	// the counter is based on block id
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint wid = tid / WORD_SZ, lid = tid % WORD_SZ;
	uint leader_lid = __ffs(__ballot(1)) - 1;
	uint icounter = wid & (NCOUNTERS - 1);
	uint cv;
	if(lid == leader_lid)
		cv = atomicAdd(counters_g + icounter, COUNTER_INC);
	cv = __shfl((int)counter_val, leader_lid);
	void *p = 0;
	// initial position
	// TODO: use a real but cheap random number generator
	uint iblock = (tid * THREAD_FREQ + cv * cv * (cv + 1)) & 
		(size_infos_g[size_id].nblocks - 1);
	// main allocation loop
	bool want_alloc = true;
	// use two-level loop to avoid warplocks
	do {
		if(want_alloc) {
			// try allocating in head superblock
			uint head_sb = head_sbs_g[size_id];
			p = sb_alloc_in(head_sb, &iblock, size_id);
			bool need_roomy_sb = want_alloc = !p;
			uint need_roomy_mask;
			while(need_roomy_mask = __ballot(need_roomy_sb)) {
				if(need_roomy_sb) {
					leader_lid = __ffs(need_roomy_mask) - 1;
					uint leader_size_id = __shfl((int)size_id, leader_lid);
					// here try to check whether a new SB is really needed, and get the
					// new SB
					if(lid == leader_lid)
						head_sb = new_sb_for_size(size_id);
					if(size_id == __shfl((int)leader_size_id, leader_lid)) {
						head_sb = __shfl((int)head_sb, leader_lid);
						want_alloc = head_sb != SB_NONE;
						need_roomy_sb = false;
					}
				}
			}  // while(need new head superblock)
		}
	} while(__any(want_alloc));
	if(!p)
		printf("cannot allocate memory\n");
	return p;
}  // hamalloc

__device__ void hafree(void *p) {
	// ignore zero pointer
	if(!p)
		return;
	// get the cell descriptor and other data
	uint icell;
	uint64 cell = grid_cell(p, &icell);
	uint size_id = grid_size_id(icell, cell, p);
	uint sb_id = grid_sb_id(icell, cell, p);
	uint *bits = sb_block_bits(sb_id);
	// free the memory
	uint iblock = ((char *)p - (char *)sbs_g[sb_id].ptr) / 
		size_infos_g[size_id].block_sz;
	uint iword = iblock / WORD_SZ, ibit = iblock % WORD_SZ;
	uint new_word = atomicAnd(block_bits_g + iword, ~(1 << ibit)) & 
		~(1 << ibit);
	sb_dctr_dec(size_id, sb_id, new_word, iword);
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
	uint nsbs = dev_memory / SB_SZ;
	void *nsbs_addr;
	cucheck(cudaGetSymbolAddress(&nsbs_addr, nsbs_g));
	cucheck(cudaMemcpy(nsbs_addr, &nsbs, sizeof(uint), cudaMemcpyHostToDevice));

	// allocate a fixed number of superblocks, copy them to device
	uint nsbs_alloc = min(nsbs, NSBS_ALLOC);
	size_t sbs_sz = MAX_NSBS * sizeof(superblock_t);
	superblock_t *sbs = (superblock_t *)malloc(sbs_sz);
	memset(sbs, 0, MAX_NSBS * sizeof(superblock_t));
	for(uint isb = 0; isb < nsbs_alloc; isb++) {
		sbs[isb].noccupied = 0;
		sbs[isb].id = isb;
		cucheck(cudaMalloc(&sbs[isb].ptr, SB_SZ));
	}
	void *sbs_addr;
	cucheck(cudaGetSymbolAddress(&sbs_addr, sbs_g));
	cucheck(cudaMemcpy(sbs_addr, sbs, sbs_sz, cudaMemcpyHostToDevice));

	// allocate block bits and zero them out
	void *bit_blocks, nsb_bit_words_addr;
	uint nsb_bit_words = SB_SZ / BLOCK_STEP;
	cucheck(cudaGetSymbolAddress(&nsb_bit_words_addr, nsb_bit_words_g));
	cucheck(cudaMemcpy(nsb_bit_words_addr, &nsb_bit_words, sizeof(uint), 
										 cudaMemcpyHostToDevice));
	size_t bit_blocks_sz = nsb_bit_words * nsbs;
	cucheck(cudaMalloc(&bit_blocks, bit_blocks_sz));
	cucheck(cudaMemset(bit_blocks, 0, bit_blocks_sz));
	void *bit_blocks_addr;
	cucheck(cudaGetSymbolAddress(&bit_blocks_addr, bit_blocks_g));
	cucheck(cudaMemcpy(bit_blocks_addr, &bit_blocks, sizeof(void *), 
										 cudaMemcpyHostToDevice));
	// FINISHED HERE
	// TODO: finish initialization
	
	// zero out what need to be zeroed out

	// free all temporary data structures
	free(sbs);
}  // ha_init

void ha_shutdown(void) {
	// TODO: free memory
}  // ha_shutdown
