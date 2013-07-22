#include "utils.h"
#include "size-info.h"

/** real possible number of superblocks (based on device memory and superblock
		size) */
__constant__ uint nsbs_g;
/** block bits for all superblocks (content may change) */
uint * __constant__ block_bits_g;
/** allocation size id's (content may change); size id for each allocation takes
		1 byte, with size id "all set" meaning "no allocation" */
uint * __constant__ alloc_sizes_g;
/** number of block bit words per superblock */
__constant__ uint nsb_bit_words_g;
/** shift for number of words per superblock */
__constant__ uint nsb_bit_words_sh_g;
/** number of alloc sizes per superblock */
__constant__ uint nsb_alloc_words_g;
/** superblock size (common for all superblocks, power-of-two) */
__constant__ uint sb_sz_g;
/** superblock size shift (for fast division operations) */
__constant__ uint sb_sz_sh_g;

/** superblock descriptors */
__device__ superblock_t sbs_g[MAX_NSBS];
/** superblock's (non-distributed) counters */
__device__ uint sb_counters_g[MAX_NSBS];

/** the set of all free superblocks */
__device__ sbset_t free_sbs_g;
/** the set of "roomy" superblocks for the size */
__device__ sbset_t roomy_sbs_g[MAX_NSIZES];

/** head superblocks for each size */
__device__ uint head_sbs_g[MAX_NSIZES];
/** superblock operation locks per size */
__device__ uint head_locks_g[MAX_NSIZES];

/** gets block bits for superblock */
__device__ inline uint *sb_block_bits(uint sb) {
	// TODO: use a shift constant
	//return block_bits_g + sb * nsb_bit_words_g;
	return block_bits_g + (sb << nsb_bit_words_sh_g);
}  // sb_block_bits

/** gets the alloc sizes for the superblock */
__device__ inline uint *sb_alloc_sizes(uint sb) {
	// TODO: use a shift constant
	return alloc_sizes_g + sb * nsb_alloc_words_g;
}

/** sets allocation size for the allocation 
		@param alloc_words allocation data for this superblock
		@param ichunk the first allocated chunk
		@param size_id the size id of the allocation
 */
__device__ inline void sb_set_alloc_size(uint *alloc_words, uint ichunk, uint
		size_id) {
	uint iword = ichunk / 4, ibyte = ichunk % 4, shift = ibyte * 8;
	uint mask = (size_id << shift) | (~0 ^ (0xffu << shift));
	atomicAnd(&alloc_words[iword], mask);
}  // sb_set_alloc_size

/** gets (and resets) allocation size for the allocation */
__device__ inline uint sb_get_reset_alloc_size(uint *alloc_words, uint ichunk) {
	uint iword = ichunk / 4, ibyte = ichunk % 4, shift = ibyte * 8;
	uint mask = 0xffu << shift;
	return (atomicOr(&alloc_words[iword], mask) >> shift) & 0xffu;
}  // sb_get_reset_alloc_size

/** tries to mark a slab as free 
		@param from_head whether there's a try to mark slab as free during detaching
		from head (this is very unlikely)
 */
__device__ inline void sb_try_mark_free(uint sb, uint size_id, bool from_head) {
	// try marking slab as free
	uint old_counter = sb_counter_val(0, false, SZ_NONE, size_id);
	uint new_counter = sb_counter_val(0, false, SZ_NONE, SZ_NONE);
	if(atomicCAS(&sb_counters_g[sb], old_counter, new_counter)) {
		// slab marked as free, remove it from roomy and add to free
		if(!from_head)
			sbset_remove_from(roomy_sbs_g[size_id], sb);
		sbs_g[sb].size_id = SZ_NONE;
		sbset_add_to(free_sbs_g, sb);
	} else if(from_head) {
		// add it to non-free
		sbset_add_to(roomy_sbs_g[size_id], sb);
	}
}  // sb_try_mark_free

/** increment the non-distributed counter of the superblock 
		size_id is just ignored, allocation size is expressed in chunks
		@returns old counter value
 */
__device__ __forceinline__ bool sb_ctr_inc
(uint size_id, uint sb_id, uint alloc_sz) {
	bool want_inc = true;
	uint mask, old_counter, lid = threadIdx.x % WARP_SZ;
	while(mask = __ballot(want_inc)) {
		uint leader_lid = warp_leader(mask), leader_sb_id = sb_id;
		leader_sb_id = __shfl((int)leader_sb_id, leader_lid);
		// allocation size is same for all superblocks
		//uint change = alloc_sz * __popc(__ballot(sb_id == leader_sb_id));
		uint change = __popc(__ballot(sb_id == leader_sb_id));
		if(lid == leader_lid)
			old_counter = sb_counter_inc(&sb_counters_g[sb_id], change);
		if(leader_sb_id == sb_id)
			old_counter = __shfl((int)old_counter, leader_lid);
		want_inc = want_inc && sb_id != leader_sb_id;
	}  // while
	//return true;
	return sb_size_id(old_counter) == size_id;
}  // sb_ctr_inc

/** increment the non-distributed counter of the superblock 
		size_id is just ignored, allocation size is expressed in chunks
 */
__device__ __forceinline__ void sb_ctr_dec
(uint size_id, uint sb_id, uint alloc_sz) {
	bool want_inc = true;
	uint mask, lid = threadIdx.x % WARP_SZ;
	while(mask = __ballot(want_inc)) {
		uint leader_lid = warp_leader(mask), leader_sb_id = sb_id;
		leader_sb_id = __shfl((int)leader_sb_id, leader_lid);
		// allocation size is same for all superblocks
		//uint change = alloc_sz * __popc(__ballot(sb_id == leader_sb_id));
		uint change = __popc(__ballot(sb_id == leader_sb_id));
		if(lid == leader_lid) {
			uint old_counter = sb_counter_dec(&sb_counters_g[sb_id], change);
			if(!sb_is_head(old_counter)) {
				// slab is non-head, so do manipulations
				uint old_count = sb_count(old_counter), new_count = old_count - change;
				//if((old_count + SB_FREE_STEP - 1) % SB_FREE_STEP != 
				//	 (new_count + SB_FREE_STEP - 1) % SB_FREE_STEP) {					
				uint threshold = size_infos_g[size_id].roomy_threshold;
				if(new_count <= threshold && old_count > threshold && new_count > 0) {
					// mark superblock as roomy for current size
					sbset_add_to(roomy_sbs_g[size_id], sb_id);
				} else if(new_count == 0) {
					sb_try_mark_free(sb_id, size_id, false);
				}  // if(slab position in sets changes)
				// }
			}  // if(not a head slab) 
		} // if(leader lane)
		want_inc = want_inc && sb_id != leader_sb_id;
	}  // while(any one wants to deallocate)
}  // sb_ctr_dec

/** tries to find a new superblock for the given size
		@returns the new head superblock id if found, and SB_NONE if none
		@remarks this function should be called by at most 1 thread in a warp at a time
*/
__device__ inline uint new_sb_for_size(uint size_id) {
	// try locking size id
	// TODO: make those who failed to lock attempt to allocate
	// in what free space left there
	//uint64 t1 = clock64();
	uint cur_head = *(volatile uint *)&head_sbs_g[size_id];
	if(try_lock(&head_locks_g[size_id])) {
		// locked successfully, check if really need replacing blocks
		uint new_head = SB_NONE;
		uint roomy_threshold = size_infos_g[size_id].roomy_threshold;
		if(cur_head == SB_NONE || 
			 sb_count(*(volatile uint *)&sb_counters_g[cur_head]) >=
			 size_infos_g[size_id].busy_threshold) {
			// replacement really necessary; first try among roomy sb's of current 
			// size
			while((new_head = sbset_get_from(roomy_sbs_g[size_id]))
						!= SB_NONE) {
				// try set head
				uint old_counter = atomicOr(&sb_counters_g[new_head], 1 << SB_HEAD_POS);
				if(sb_is_head(old_counter)) { 
				} else if(sb_size_id(old_counter) != size_id 
									|| sb_count(old_counter)	> size_infos_g[size_id].roomy_threshold) {
					// drop the block and go for another
					// TODO: process this as another head detachment
					atomicAnd(&sb_counters_g[new_head], ~(1 << SB_HEAD_POS));
				} else
					break;
			}  // while(searching through new heads)

			if(new_head == SB_NONE) {
				// try getting from free superblocks; hear actually getting one 
				// always means success, as only truly free block get to this bit array
				new_head = sbset_get_from(free_sbs_g);
				if(new_head != SB_NONE) {
					// fill in the slab
					*(volatile uint *)&sbs_g[new_head].size_id = size_id;
					uint old_counter = sb_counter_val(0, false, SZ_NONE, SZ_NONE);
					uint new_counter = sb_counter_val(0, true, SZ_NONE, size_id);
					// there may be others trying to set the head; as they come from
					// roomy blocks, they will fail; also, there may be some ongoing
					// allocation attempts, so just wait
					while(atomicCAS(&sb_counters_g[new_head], old_counter, new_counter) !=
								old_counter);
					//atomicCAS(&sb_counters_g[new_head], old_counter, new_counter);
				}  // if(got new head from free slabs)
			}  // if(didn't get new head from roomy slabs)
			if(new_head != SB_NONE) {
				// set the new head, so that allocations can continue
				head_sbs_g[size_id] = new_head;
				__threadfence();
				// detach current head
				if(cur_head != SB_NONE) {
					uint old_counter = atomicAnd(&sb_counters_g[cur_head], 
																			 ~(1 << SB_HEAD_POS));
					uint count = sb_count(old_counter);
					if(count == 0) {
						// very unlikely
						sb_try_mark_free(cur_head, size_id, true);
					} else if(count <= roomy_threshold) {
						// mark as roomy
						sbset_add_to(roomy_sbs_g[size_id], cur_head);
					} 
				}  // if(there's a head to detach)
			}  // if(found new head)
		} else {
			// just re-read the new head superblock
			new_head = *(volatile uint *)&head_sbs_g[size_id];
		}
		unlock(&head_locks_g[size_id]);
		//uint64 t2 = clock64();
		//printf("needed %lld cycles to find new head slab\n", t2 - t1);
		//printf("new head = %d\n", new_head);
		return new_head;
	} else {
		// someone else working on current head superblock; 
		while(true) {
			if(*(volatile uint *)&head_sbs_g[size_id] != cur_head ||
				 *(volatile uint *)&head_locks_g[size_id] == 0)
				break;
		}
		return *(volatile uint *)&head_sbs_g[size_id];
	}
}  // new_sb_for_size

	/** allocates memory inside the superblock 
			@param isb the superblock inside which to allocate
			@param [in,out] iblock the block from which to start searching
			@param size_id the size id for the allocation
			@returns the pointer to the allocated memory, or 0 if unable to allocate
	*/
	__device__ __forceinline__ void *sb_alloc_in
		(uint isb, uint &iblock, size_info_t size_info, uint size_id) {
		if(isb == SB_NONE)
			return 0;
		void *p = 0;
		uint *block_bits = sb_block_bits(isb);
		superblock_t sb = sbs_g[isb];
		// check the superblock occupancy counter
		uint sb_counter = sb_counters_g[isb];
		if(sb_count(sb_counter) >= size_info.busy_threshold) {
			uint count = sb_count(*(volatile uint *)&sb_counters_g[isb]);
			//uint count = sb_count(sb_counter);
			// try allocate nevertheless if head is locked
			if(count >= size_info.nblocks || !*(volatile uint *)&head_locks_g[size_id])
				return 0;
		}
		uint iword, ibit;
		bool reserved = false;
		// iterate until successfully reserved
		for(uint itry = 0; itry < MAX_NTRIES; itry++) {
			// try reserve
			iword = iblock / WORD_SZ;
			ibit = iblock % WORD_SZ;
			uint alloc_mask = 1 << ibit;
			uint old_word = atomicOr(block_bits + iword, alloc_mask);
			if(!(old_word & alloc_mask)) {
				// initial reservation successful
				reserved = true;
				break;
			} else {
				iblock = (iblock + size_info.hash_step) % size_info.nblocks;
			}
		}
		if(reserved) {
			// increment counter
			if(!sb_ctr_inc(size_id, isb, 1)) {
				// reservation unsuccessful (slab was freed), cancel it
				sb_counter_dec(&sb_counters_g[isb], 1);
				atomicAnd(block_bits + iword, ~(1 << ibit));
				reserved = false;
			}
		}
		if(reserved) {
			p = (char *)sb.ptr + iblock * size_info.block_sz;
			// write allocation size
			// TODO: support chunks of other size
			//uint *alloc_sizes = sb_alloc_sizes(isb);
			//sb_set_alloc_size(alloc_sizes, iblock, size_id);
			//sb_ctr_inc(~0, isb, size_info.block_sz / BLOCK_STEP);
		}
		return p;
	}  // sb_alloc_in
