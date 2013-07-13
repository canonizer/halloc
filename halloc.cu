/** @file hamalloc.cu implementation of halloc allocator */

#include <algorithm>
#include <math.h>
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

/** the number of allocation counters*/
#define NCOUNTERS 4096
/** thread frequency for initial hashing */
#define THREAD_FREQ 17
/** allocation counter increment */
#define COUNTER_INC 1
/** word size (the word is uint, which is assumed to be 32-bit) */
#define WORD_SZ 32
/** the warp size (32 on current NVidia architectures) */
#define WARP_SZ 32
/** maximum number of superblocks */
#define MAX_NSBS 4096
/** number of superblocks to allocate initially */
//#define NSBS_ALLOC 64
/** maximum amount of memory to allocate, in MB */
#define MAX_ALLOC_MEM 512
/** the size of SB set, in words; the number of used SBs can be smaller */
#define SB_SET_SZ (MAX_NSBS / WORD_SZ)
/** maximum number of sizes supported */
#define MAX_NSIZES 64
/** whether to do head check after allocation */
#define HEAD_SB_CHECK 0
/** number of superblock counters */
#define NSB_COUNTERS 16
/** a step at which to  change the value of a distributed counter */
#define SB_DISTR_STEP 16
//#define SB_DISTR_STEP_MASK (1 << 0 | 1 << 8 | 1 << 24)
/** a step to change the value of main counter */
#define SB_MAIN_STEP 512
/** a step at which to add to free superblocks */
#define SB_FREE_ADD_STEP (1 * SB_MAIN_STEP)
/** maximum number of tries inside a superblock after which the allocation
		attempt is abandoned */
#define MAX_NTRIES 64
/** a "no-sb" constant */
#define SB_NONE (~0)
/** a "no-size" constant */
#define SZ_NONE (~0)
/** block step (16 bytes by default), a power of two */
#define BLOCK_STEP 16
/** minimum block size (a power of two) */
#define MIN_BLOCK_SZ 16
/** maximum block size (a power of two) */
#define MAX_BLOCK_SZ 256
/** default superblock size, in bytes */
#define SB_SZ_SH (10 + 10 + 3)
#define SB_SZ (1 << SB_SZ_SH)

// types 

/** size information type; this is non-changing information, to be stored in
		constant memory */
typedef struct {
/** block size */
uint block_sz;
/** number of blocks in superblock */
uint nblocks;
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
	/** superblock size id */
	uint size_id;
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
void * __constant__ base_addr_g;
/** superblock size (common for all superblocks, power-of-two) */
__constant__ uint sb_sz_g;
/** superblock size shift (for fast division operations) */
__constant__ uint sb_sz_sh_g;

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

/** checks whether the step is in mask */
__device__ inline bool step_is_in_mask(uint mask, uint val) {
	return (mask >> val) & 1;
}

/** gets the distance to the next higher mask value	*/
__device__ inline uint step_next_dist(uint mask, uint val) {
	uint res =  __ffs(mask >> (val + 1));
	return res ? res : WORD_SZ - val;
}  

/** gets superblock from set (and removes it) */
__device__ inline uint sbset_get_from(sbset_t *sbset, uint except) {
	// TODO: maybe do several trials to be sure
	for(uint iword = 0; iword < nsbs_g / WORD_SZ; iword++)
		for(uint word = (*sbset)[iword], ibit = __ffs(word) - 1; word;
				word &= ~(1 <<	ibit), ibit = __ffs(word) - 1) {
			// try locking the bit
			if(iword * WORD_SZ + ibit != except) {
				uint mask = 1 << ibit;
				if(atomicAnd(&(*sbset)[iword], ~mask) & mask)
					return iword * WORD_SZ + ibit;
			}
		}
	return SB_NONE;
}  // sbset_get_from

/** adds ("returns") superblock to the set */
__device__ inline void sbset_add_to(sbset_t *sbset, uint sb) {
	uint iword = sb / WORD_SZ, ibit = sb % WORD_SZ;
	atomicOr(&(*sbset)[iword], 1 << ibit);
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
#define GRID_ADDR_SH 4
/** initial value for the grid cell */
__host__ __device__ inline uint64 grid_cell_init() {
	uint64 no_sb_field = (1 << GRID_SB_LEN) - 1;
	return no_sb_field << GRID_FIRST_SB_POS | no_sb_field << GRID_SECOND_SB_POS;
}
/** add the superblock to the grid 
		// TODO: use on device as well, also with size id
*/
__host__ void grid_add_sb
(uint64 *cells, void *base_addr, uint sb, void *sb_addr, uint sb_sz) {
	void *sb_end_addr = (char *)sb_addr + sb_sz - 1;
	uint icell_start = ((char *)sb_addr - (char *)base_addr) / sb_sz;
	uint icell_end = ((char *)sb_addr + sb_sz - 1 - (char *)base_addr) / sb_sz;
	for(uint icell = icell_start; icell <= icell_end; icell++) {
		uint64 cell = cells[icell];
		cell |= 1ull << GRID_INIT_POS;
		void *cell_start_addr = (char *)base_addr + (uint64)icell * sb_sz;
		void *cell_end_addr = (char *)base_addr + (uint64)(icell + 1) * sb_sz - 1;
		if(sb_addr <= cell_start_addr) {
			// set first superblock in cell
			uint64 first_sb_mask = ((1ull << GRID_SB_LEN) - 1) << GRID_FIRST_SB_POS;
			cell = ~first_sb_mask & cell | (uint64)sb << GRID_FIRST_SB_POS;
		}
		if(sb_end_addr >= cell_end_addr) {
			// set second superblock in cell
			uint64 second_sb_mask = ((1ull << GRID_SB_LEN) - 1) << GRID_SECOND_SB_POS;
			cell = ~second_sb_mask & cell | (uint64)sb << GRID_SECOND_SB_POS;
		}
		uint64 mid_addr_mask = ((1ull << GRID_ADDR_LEN) - 1) << GRID_ADDR_POS;
		// set the break address
		if(sb_addr > cell_start_addr) {
			// current superblock is the second superblock, mid address is its start
			uint64 mid_addr = ((char *)sb_addr - (char *)cell_start_addr) >> 
				GRID_ADDR_SH;
			cell = ~mid_addr_mask & cell | mid_addr << GRID_ADDR_POS;
			//printf("icell = %d, cell_addr = %p, sb_addr = %p, mid_addr = %llx\n",
			//			 icell, cell_start_addr, sb_addr, mid_addr);
		} else if(sb_end_addr <= cell_end_addr) {
			// current superblock is the first superblock, mid address is end of this
			// superblock + 1
			uint64 mid_addr = ((char *)sb_end_addr + 1 - (char *)cell_start_addr) >>
				GRID_ADDR_SH;
			cell = ~mid_addr_mask & cell | mid_addr << GRID_ADDR_POS;
			//printf("icell = %d, cell_addr = %p, sb_addr = %p, mid_addr = %llx\n",
			//			 icell, cell_start_addr, sb_addr, mid_addr);
		}
		// save the modified cell
		cells[icell] = cell;
	}  // for(each cell in interval)
}  // grid_add_sb

/** checks whether the grid cell is initialized */
__device__ inline bool grid_is_init(uint64 cell) {
	return (cell >> GRID_INIT_POS) & 1;
} 
/** gets the first size id of the grid cell */
__device__ inline uint grid_first_size_id(uint64 cell) {
	return (cell >> GRID_FIRST_SIZE_POS) & ((1ull << GRID_SIZE_LEN) - 1);
}
/** gets the  second size id of the grid cell */
__device__ inline uint grid_second_size_id(uint64 cell) {
	return (cell >> GRID_SECOND_SIZE_POS) & ((1ull << GRID_SIZE_LEN) - 1);
}
/** gets the first superblock id of the grid cell  */
__device__ inline uint grid_first_sb_id(uint64 cell) {
	return (cell >> GRID_FIRST_SB_POS) & ((1ull << GRID_SB_LEN) - 1);
}
/** gets the second superblock id of the grid cell  */
__device__ inline uint grid_second_sb_id(uint64 cell) {
	return (cell >> GRID_SECOND_SB_POS) & ((1ull << GRID_SB_LEN) - 1);
}
/** gets the mid-address of the grid cell */
__device__ inline void *grid_mid_addr(uint icell, uint64 cell) {
	uint in_sb_addr = ((cell >> GRID_ADDR_POS) & ((1ull << GRID_ADDR_LEN) - 1))
		<< GRID_ADDR_SH;
	return (char *)base_addr_g + (uint64)icell * sb_sz_g + in_sb_addr;
}
/** gets the grid cell for the pointer */
__device__ inline uint64 grid_cell(void *p, uint *icell) {
	// TODO: handle stale cell data
	//*icell = ((char *)p - (char *)base_addr_g) / sb_sz_g;
	*icell = ((char *)p - (char *)base_addr_g) >> sb_sz_sh_g;
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
	return block_bits_g + sb * nsb_bit_words_g;
}  // sb_block_bits

/** increment distributed superblock counter 
		@param size_id the size id for allocation; ignored 
		@param sb the superblock id
		@param old_word the old value of the word where allocation took place
		@param iword the index of the word where allocation took place
*/
__device__ inline void sb_dctr_inc
(uint size_id, uint sb, uint old_word, uint iword) {
	uint nword_blocks = __popc(old_word);
	if(nword_blocks % SB_DISTR_STEP == 0) {
		//if(step_is_in_mask(SB_DISTR_STEP_MASK, nword_blocks)) {
		// increment distributed counter
		//uint ictr = iword % NSB_COUNTERS;
		uint ictr = iword % NSB_COUNTERS;
		//uint old_val = atomicAdd(&sb_counters_g[sb][ictr], SB_DISTR_STEP);
		uint step = SB_DISTR_STEP;
		//uint step = step_next_dist(SB_DISTR_STEP_MASK, nword_blocks);
		uint old_val = atomicAdd(&sb_counters_g[sb][ictr], step);
		uint new_val = old_val + step;
		//if(old_val % SB_MAIN_STEP == 0) {
		if((old_val + (SB_MAIN_STEP - 1)) / SB_MAIN_STEP 
			 != (new_val + (SB_MAIN_STEP - 1)) / SB_MAIN_STEP) {
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
__device__ inline void sb_dctr_dec
(uint size_id, uint sb, uint new_word, uint iword) {
	uint nword_blocks = __popc(new_word);
	if(nword_blocks % SB_DISTR_STEP == 0) {
	//if(step_is_in_mask(SB_DISTR_STEP_MASK, nword_blocks)) {
		// decrement distributed counter
		/*
		// do thread-joint decrement of the main counter
		uint mask = __ballot(1);
		uint lid = threadIdx.x % WARP_SZ;
		uint leader_lid = __ffs(mask) - 1;
		if(lid == leader_lid) {
			uint decr = __popc(mask) * SB_MAIN_STEP;
			uint new_main_val = atomicSub(&sbs_g[sb].noccupied, decr) - 
				decr;
			if(new_main_val <= size_infos_g[size_id].roomy_threshold)
				sbset_add_to(&roomy_sbs_g[size_id], sb);
				}*/
		// decrement distributed counter
		uint ictr = iword % NSB_COUNTERS;
		uint step = SB_DISTR_STEP;
		//uint step = step_next_dist(SB_DISTR_STEP_MASK, nword_blocks);
		uint old_val = atomicSub(&sb_counters_g[sb][ictr], step);
		uint new_val = old_val - step;
		//if(new_val % SB_MAIN_STEP == 0) {
		if((new_val + (SB_MAIN_STEP - 1)) / SB_MAIN_STEP != 
			 (old_val + (SB_MAIN_STEP - 1)) / SB_MAIN_STEP) {
			// decrement main counter
			uint new_main_val = atomicSub(&sbs_g[sb].noccupied, SB_MAIN_STEP) - 
				SB_MAIN_STEP;
			if(new_main_val <= size_infos_g[size_id].roomy_threshold) {
				// mark superblock as roomy for current size
				//printf("size_id = %d, sb = %d\n", size_id, sb);
				sbset_add_to(&roomy_sbs_g[size_id], sb);
			}
		}
	}
}  // sb_dctr_dec

/** allocates memory inside the superblock 
		@param isb the superblock inside which to allocate
		@param [in,out] iblock the block from which to start searching
		@param size_id the size id for the allocation
		@returns the pointer to the allocated memory, or 0 if unable to allocate
*/
__device__ inline void *sb_alloc_in(uint isb, uint &iblock, size_info_t size_info) {
	if(isb == SB_NONE)
		return 0;
	void *p = 0;
	uint *block_bits = sb_block_bits(isb);
	superblock_t sb = sbs_g[isb];
	//size_info_t size_info = size_infos_g[size_id];
	// check the superblock occupancy counter
	// a volatile read doesn't really harm
	//uint noccupied = *(volatile uint *)&sbs_g[sb].noccupied;
	if(sb.noccupied >= size_info.busy_threshold)
		return 0;
	uint old_word, iword;
	bool reserved = false;
	// iterate until successfully reserved
	for(uint itry = 0; itry < MAX_NTRIES; itry++) {
		//for(uint i = 0; i < 1; i++) {
		// try reserve
		iword = iblock / WORD_SZ;
		uint ibit = iblock % WORD_SZ, alloc_mask = 1 << ibit;
		old_word = atomicOr(block_bits + iword, alloc_mask);
		if(!(old_word & alloc_mask)) {
			// reservation successful
			reserved = true;
			break;
		} else {
			iblock = (iblock + size_info.hash_step) % size_info.nblocks;
			//iblock = (iblock + size_infos_g[size_id].hash_step) 
			//	& (size_infos_g[size_id].nblocks - 1);
		}
	}
	if(reserved) {
		p = (char *)sb.ptr + iblock * size_info.block_sz;
		sb_dctr_inc(~0, isb, old_word, iword);
	}
	//printf("sbs_g[%d].ptr = %p, allocated p = %p\n", sb, sbs_g[sb].ptr, p);
	return p;
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
		uint new_head = SB_NONE;
		if(cur_head == SB_NONE || 
			*(volatile uint *)&sbs_g[cur_head].noccupied >=
			size_infos_g[size_id].busy_threshold) {
			// replacement really necessary; first try among roomy sb's of current 
			// size
			new_head = sbset_get_from(&roomy_sbs_g[size_id], cur_head);
			if(new_head == SB_NONE) {
				// try getting from free superblocks
				new_head = sbset_get_from(&free_sbs_g, SB_NONE);
			}
			if(new_head != SB_NONE) {
				// replace current head
				head_sbs_g[size_id] = new_head;
				*(volatile uint *)&sbs_g[new_head].size_id = size_id;
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
		return *(volatile uint *)&head_sbs_g[size_id];
	}
}  // new_sb_for_size

__device__ void *hamalloc(size_t nbytes) {
	// ignore zero-sized allocations
	if(!nbytes)
		return 0;
	uint size_id = (nbytes - MIN_BLOCK_SZ) / BLOCK_STEP;
	size_info_t size_info = size_infos_g[size_id];
	uint head_sb = head_sbs_g[size_id];
	// the counter is based on block id
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint wid = tid / WORD_SZ, lid = tid % WORD_SZ;
	uint leader_lid = __ffs(__ballot(1)) - 1;
	uint icounter = wid & (NCOUNTERS - 1);
	uint cv;
	if(lid == leader_lid)
		cv = atomicAdd(&counters_g[icounter], COUNTER_INC);
	cv = __shfl((int)cv, leader_lid);
	void *p = 0;
	// initial position
	// TODO: use a real but cheap random number generator
	// uint cv2 = cv >> 4, cv1 = cv & 15;
	// uint iblock = (tid * THREAD_FREQ + cv1 + cv2 * cv2 * (cv2 + 1)) %
	//  	size_infos_g[size_id].nblocks;
	uint iblock = (tid * THREAD_FREQ + cv * cv * (cv + 1)) % size_info.nblocks;
	//uint iblock = (tid * THREAD_FREQ + cv * cv * (cv + 1)) &
	//	(size_infos_g[size_id].nblocks - 1);
	// main allocation loop
	bool want_alloc = true;
	//uint head_sb;
	// use two-level loop to avoid warplocks
	do {
		if(want_alloc) {
			// try allocating in head superblock
			//head_sb = head_sbs_g[size_id];
			p = sb_alloc_in(head_sb, iblock, size_info);
			bool need_roomy_sb = want_alloc = !p;
			while(__any(need_roomy_sb)) {
				uint need_roomy_mask = __ballot(need_roomy_sb);
				if(need_roomy_sb) {
					leader_lid = __ffs(need_roomy_mask) - 1;
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
		}
	} while(__any(want_alloc));
	//} while(want_alloc);
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
	uint size_id = sbs_g[sb_id].size_id;
	uint *block_bits = sb_block_bits(sb_id);
	// free the memory
	uint iblock = (uint)((char *)p - (char *)sbs_g[sb_id].ptr) / 
		size_infos_g[size_id].block_sz;
	uint iword = iblock / WORD_SZ, ibit = iblock % WORD_SZ;
	uint new_word = atomicAnd(block_bits + iword, ~(1 << ibit)) & ~(1 << ibit);
	//printf("freeing: sb_id = %d, p = %p, iblock = %d\n", sb_id, p, iblock);
	sb_dctr_dec(size_id, sb_id, new_word, iword);
}  // hafree

/** find the largest prime number below this one, and not dividing this one */
uint max_prime_below(uint n) {
	for(uint p = n - 1; p >= 3; p--) {
		uint max_d = (uint)floor(sqrt(p));
		bool is_prime = true;
		for(uint d = 2; d <= max_d; d++)
			if(p % d == 0) {
				is_prime = false;
				break;
			}
		if(is_prime && n % p)
			return p;
	}
	// if we are here, we can't find prime; exit with failure
	fprintf(stderr, "cannot find prime below %d not dividing %d\n", n, n);
	exit(-1);
	return ~0;
}  // max_prime_below

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
	char *base_addr = (char *)~0ull;
	for(uint isb = 0; isb < nsbs_alloc; isb++) {
		sbs[isb].noccupied = 0;
		sbs[isb].size_id = SZ_NONE;
		cucheck(cudaMalloc(&sbs[isb].ptr, SB_SZ));
		base_addr = (char *)min((uint64)base_addr, (uint64)sbs[isb].ptr);
	}
	//cuset_arr(sbs_g, (superblock_t (*)[MAX_NSBS])&sbs);
	cuset_arr(sbs_g, (superblock_t (*)[MAX_NSBS])sbs);
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
	void *bit_blocks;
	uint nsb_bit_words = SB_SZ / BLOCK_STEP;
	cuset(nsb_bit_words_g, uint, nsb_bit_words);
	size_t bit_blocks_sz = nsb_bit_words * nsbs;
	cucheck(cudaMalloc(&bit_blocks, bit_blocks_sz));
	cucheck(cudaMemset(bit_blocks, 0, bit_blocks_sz));
	cuset(block_bits_g, uint *, (uint *)bit_blocks);

	// set sizes info
	uint nsizes = (MAX_BLOCK_SZ - MIN_BLOCK_SZ) / BLOCK_STEP + 1;
	cuset(nsizes_g, uint, nsizes);
	size_info_t size_infos[MAX_NSIZES];
	memset(size_infos, 0, MAX_NSIZES * sizeof(size_info_t));
	for(uint isize = 0; isize < nsizes; isize++) {
		size_info_t *size_info = &size_infos[isize];
		size_info->block_sz = MIN_BLOCK_SZ + BLOCK_STEP * isize;
		size_info->nblocks = sb_sz / size_info->block_sz;
		size_info->hash_step = 
			max_prime_below(size_info->nblocks / 256 + size_info->nblocks / 64);
		// size_info->hash_step = 
		// 	max_prime_below(size_info->nblocks / 128);
		//size_info->hash_step = size_info->nblocks / 256 + size_info->nblocks / 64 - 1;
		size_info->roomy_threshold = 0.4 * size_info->nblocks;
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
	cuvar_memset(unallocated_sbs_g, 0, sizeof(unallocated_sbs_g));
	cuvar_memset(roomy_sbs_g, 0, sizeof(roomy_sbs_g));
	cuvar_memset(head_sbs_g, ~0, sizeof(head_sbs_g));
	cuvar_memset(head_locks_g, 0, sizeof(head_locks_g));
	cuvar_memset(sb_counters_g, 0, sizeof(sb_counters_g));
	cuvar_memset(counters_g, 1, sizeof(counters_g));

	// free all temporary data structures
	free(sbs);

	/** ensure that all CUDA stuff has finished */
	cucheck(cudaStreamSynchronize(0));
}  // ha_init

void ha_shutdown(void) {
	// TODO: free memory
}  // ha_shutdown
