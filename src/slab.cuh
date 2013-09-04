#include "utils.h"
#include "size-info.h"

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

/** superblock descriptors */
__device__ superblock_t sbs_g[MAX_NSBS];
/** superblock's (non-distributed) counters */
__device__ uint sb_counters_g[MAX_NSBS];

/** the set of all unallocated slabs */
__device__ sbset_t unallocated_sbs_g;
/** the set of all free slabs */
__device__ sbset_t free_sbs_g;
/** the set of "sparse" slabs for the size */
__device__ sbset_t sparse_sbs_g[MAX_NCHUNK_IDS];
/** the set of "roomy" slabs for the size */
__device__ sbset_t roomy_sbs_g[MAX_NSIZES];

/** head superblocks for each size */
__device__ uint head_sbs_g[NHEADS][MAX_NSIZES];
/** cached head SB's for each size */
__device__ volatile uint cached_sbs_g[NHEADS][MAX_NSIZES];
/** superblock operation locks per size */
__device__ uint head_locks_g[NHEADS][MAX_NSIZES];

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

__device__ uint dummy_g;

/** sets allocation size for the allocation 
		@param alloc_words allocation data for this superblock
		@param ichunk the first allocated chunk
		@param nchunks the number of chunks allocated, max 15
		@param size_id id of the allocation size
 */
__device__ inline void sb_set_alloc_size
(uint *alloc_words, uint ichunk, uint	nchunks) {
	uint iword = ichunk / 4, ibyte = ichunk % 4, shift = ibyte * 8;
	uint mask = nchunks << shift;
	//dummy_g += atomicOr(&alloc_words[iword], mask);
	atomicOr(&alloc_words[iword], mask);
	//uint mask = (size_id << shift) | (~0 ^ (0xfu << shift));
	//atomicAnd(&alloc_words[iword], mask);
}  // sb_set_alloc_size

/** gets (and resets) allocation size for the allocation 
		@returns the number of chunks allocated for this allocation (max 15)
 */
__device__ inline uint sb_get_reset_alloc_size(uint *alloc_words, uint ichunk) {
	uint iword = ichunk / 4, ibyte = ichunk % 4, shift = ibyte * 8;
	uint mask = ~(0xf << shift);
	return (atomicAnd(&alloc_words[iword], mask) >> shift) & 0xfu;
	//uint mask = 0xfu << shift;
	//return (atomicOr(&alloc_words[iword], mask) >> shift) & 0xfu;
}  // sb_get_reset_alloc_size

/** tries to mark a slab as free 
		@param from_head whether there's a try to mark slab as free during detaching
		from head (this is very unlikely)
 */
__device__ inline void sb_try_mark_free
(uint sb, uint size_id, uint chunk_id, bool from_head) {
	// try marking slab as free
	uint old_counter = sb_counter_val(0, false, chunk_id, size_id);
	uint new_counter = sb_counter_val(0, false, SZ_NONE, SZ_NONE);
	if(atomicCAS(&sb_counters_g[sb], old_counter, new_counter) == old_counter) {
		//if(atomicAdd(&sb_counters_g[sb], new_counter - old_counter) == old_counter) {
		//printf("slab %d free, ctr = %x\n", sb, old_counter);
		// slab marked as free, remove it from roomy and add to free
		if(!from_head)
			sbset_remove_from(sparse_sbs_g[chunk_id], sb);
		sbs_g[sb].size_id = SZ_NONE;
		sbs_g[sb].chunk_sz = 0;
		sbset_add_to(free_sbs_g, sb);
	} else if(from_head) {
		// TODO: also check for adding to sparse slabs
		// add it to non-free
		//atomicSub(&sb_counters_g[sb], new_counter - old_counter);
		sbset_add_to(roomy_sbs_g[size_id], sb);
	}
}  // sb_try_mark_free

/** increment the non-distributed counter of the superblock 
		size_id is just ignored, allocation size is expressed in chunks
		@returns a mask which indicates
		bit 0 = whether allocation succeeded (0 = allocation failed due to block
		being freed)
		bit 1 = whether size_id threshold have been crossed 
 */
__device__ __forceinline__ uint sb_ctr_inc
(uint size_id, uint chunk_id, uint sb_id, uint nchunks) {
	uint return_mask = 1;
	bool want_inc = true;
	uint mask, old_counter, lid = lane_id(), leader_lid;
	uint change;
	//while(mask = __ballot(want_inc)) {
	//	if(want_inc) {
	while(want_inc) {
		mask = __ballot(1);
		leader_lid = warp_leader(mask);
		uint leader_sb_id = sb_id;
		leader_sb_id = __shfl((int)leader_sb_id, leader_lid);
		// allocation size is same for all superblocks
		uint group_mask = __ballot(sb_id == leader_sb_id);
		change = nchunks * __popc(group_mask);
		if(lid == leader_lid)
			old_counter = sb_counter_inc(&sb_counters_g[sb_id], change);
		//mask &= ~group_mask;
		want_inc = want_inc && sb_id != leader_sb_id;
		//}
	}  // while
	old_counter = __shfl((int)old_counter, leader_lid);
	if(sb_chunk_id(old_counter) != chunk_id)
		return_mask &= ~1;
	uint old_count = sb_count(old_counter);
	uint threshold = size_infos_g[size_id].busy_threshold;
	if(old_count + change >= threshold && old_count < threshold)
		return_mask |= 2;
	return return_mask;
}  // sb_ctr_inc

/** increment the non-distributed counter of the superblock 
		sb_id id of the slab for which the counter is decremented
		alloc_sz allocation size decrement for the thread (in chunks)
 */
__device__ __forceinline__ void sb_ctr_dec(uint sb_id, uint nchunks) {
	bool want_inc = true;
	uint mask, lid = lane_id();
	while(mask = __ballot(want_inc)) {
		uint leader_lid = warp_leader(mask), leader_sb_id = sb_id;
		uint leader_nchunks = nchunks;
		leader_sb_id = __shfl((int)leader_sb_id, leader_lid);
		leader_nchunks = __shfl((int)leader_nchunks, leader_lid);
		// allocation size is same for all superblocks
		// TODO: handle the situation when different allocation sizes are 
		// freed within the same slab, and do reduction for that
		bool want_now = sb_id == leader_sb_id && nchunks == leader_nchunks;
		uint change = nchunks * __popc(__ballot(want_now));
		if(lid == leader_lid) {
			uint old_counter = sb_counter_dec(&sb_counters_g[sb_id], change);
			if(!sb_is_head(old_counter)) {
				uint size_id = sb_size_id(old_counter);
				// slab is non-head, so do manipulations
				uint old_count = sb_count(old_counter), new_count = old_count - change;
				uint roomy_threshold = size_infos_g[size_id].roomy_threshold;
				if(new_count <= roomy_threshold && old_count > roomy_threshold) {
					// mark superblock as roomy for current size
					sbset_add_to(roomy_sbs_g[size_id], sb_id);
				} else {
					uint sparse_threshold = size_infos_g[size_id].sparse_threshold;
					uint chunk_id = sb_chunk_id(old_counter);
					if(new_count <= sparse_threshold && 
						 old_count > sparse_threshold)	{
						sbset_remove_from(roomy_sbs_g[size_id], sb_id);
						sbset_add_to(sparse_sbs_g[chunk_id], sb_id);
					}	else if(new_count == 0) {
						sb_try_mark_free(sb_id, size_id, chunk_id, false);
					}  // if(slab position in sets changes)
				}
			}  // if(not a head slab) 
		} // if(leader lane)
		want_inc = want_inc && !want_now;
	}  // while(any one wants to deallocate)
}  // sb_ctr_dec

/** detaches the specified head slab */
__device__ __forceinline__ void detach_head(uint head) {
	uint old_counter = atomicAnd(&sb_counters_g[head], 
															 ~(1 << SB_HEAD_POS));
	uint count = sb_count(old_counter);
	uint size_id = sb_size_id(old_counter);
	// TODO: specialize for head detachment vs. stale search metadata; this
	// conditional is only necessary in the latter case
	if(size_id == (SZ_NONE & ((1 << SB_SIZE_SZ) - 1))) {
		// unconditionally add to free		
		//printf("unconditionally adding slab %d to free\n", head);
		sbset_add_to(free_sbs_g, head);
	} else {
		uint roomy_threshold = size_infos_g[size_id].roomy_threshold;
		if(count <= roomy_threshold) {
			uint chunk_id = sb_chunk_id(old_counter);
			if(count == 0) {
				sb_try_mark_free(head, size_id, chunk_id, true);
			} else {
				uint sparse_threshold = size_infos_g[chunk_id].sparse_threshold;
				if(count <= sparse_threshold) {
					sbset_add_to(sparse_sbs_g[chunk_id], head);
				} else {
					sbset_add_to(roomy_sbs_g[size_id], head);
				}
			}  // if(free)
		}  // if(at least roomy)
	}  // if(size_id == SZ_NONE)
}  // detach_head

/** finds a suitable new slab for size and just returns it, without modifying
		any of the underlying size data structures */
__device__ __forceinline__ uint find_sb_for_size(uint size_id, uint chunk_id) {
	uint new_head = SB_NONE;
	// TODO: ensure that not checking counts against thresholds really 
	// doesn't hurt ;)
	// TODO: maybe several trials, as sets tend to be only eventually consistent
	// first try among roomy sb's of current size
	while((new_head = sbset_get_from(roomy_sbs_g[size_id])) != SB_NONE) {
		// try set head
		uint old_counter = atomicOr(&sb_counters_g[new_head], 1 << SB_HEAD_POS);
		if(sb_is_head(old_counter)) { 
		} else if(sb_size_id(old_counter) != size_id) {
			//							|| sb_count(old_counter) > size_infos_g[size_id].roomy_threshold) {
			// drop the block and go for another
			// TODO: process this as another head detachment
			//atomicAnd(&sb_counters_g[new_head], ~(1 << SB_HEAD_POS));
			detach_head(new_head);
		} else
			break;
	}  // while(searching through new heads)

	// try getting from sparse slabs	
	if(new_head == SB_NONE) {
		while((new_head = sbset_get_from(sparse_sbs_g[chunk_id])) != SB_NONE) {
			// try set head
			uint old_counter = atomicOr(&sb_counters_g[new_head], 1 << SB_HEAD_POS);
			if(sb_is_head(old_counter)) { 
			} else if(sb_chunk_id(old_counter) != chunk_id) {
								// || sb_count(old_counter) > size_infos_g[size_id].sparse_threshold) {
				// drop the block and go for another
				// TODO: process this as another head detachment
				//atomicAnd(&sb_counters_g[new_head], ~(1 << SB_HEAD_POS));
				detach_head(new_head);
			} else {
				// almost there, but still need to set size
				// TODO: check that this causes no race conditions
				uint old_size_id = sb_size_id(old_counter) & ((1 << SB_SIZE_SZ) - 1);
				atomicXor(&sb_counters_g[new_head], old_size_id ^ size_id);
				*(volatile uint *)&sbs_g[new_head].size_id = size_id;
				break;
			}
		}  // while(searching through new heads)
	}

	// try getting from free slabs; hear actually getting one 
	// always means success, as only truly free block get to this bit array
	if(new_head == SB_NONE) {
		while((new_head = sbset_get_from(free_sbs_g)) != SB_NONE) {
			//if(new_head != SB_NONE) {
			// fill in the slab
			uint old_counter = sb_counter_val(0, false, SZ_NONE, SZ_NONE);
			uint new_counter = sb_counter_val(0, true, chunk_id, size_id);
			// there may be others trying to set the head; as they come from
			// roomy blocks, they will fail; also, there may be some ongoing
			// allocation attempts, so just wait
			//printf("waiting for slab %d to be fully free\n", new_head);
			uint real_old_counter;
			if((real_old_counter = atomicCAS
					(&sb_counters_g[new_head], old_counter, new_counter)) ==
				 old_counter) {
				*(volatile uint *)&sbs_g[new_head].size_id = size_id;
				*(volatile uint *)&sbs_g[new_head].chunk_sz = 
					size_infos_g[size_id].chunk_sz;
				break;
			} else {
				//printf("cannot get free block %d, counter %x\n", new_head, real_old_counter);
			}
			//printf("the slab %d is free\n", new_head);
			//atomicCAS(&sb_counters_g[new_head], old_counter, new_counter);
		}  // if(got new head from free slabs)
	}
	
	return new_head;
	// TODO: request additional memory from CUDA allocator
}  // find_sb_for_size

/** tries to find a new superblock for the given size
		@returns the new head superblock id if found, and SB_NONE if none
		@remarks this function should be called by at most 1 thread in a warp at a time
*/
__device__ __forceinline__ uint new_sb_for_size
(uint size_id, uint chunk_id, uint ihead) {
	// try locking size id
	// TODO: make those who failed to lock attempt to allocate
	// in what free space left there
	//uint64 t1 = clock64();
	uint cur_head = *(volatile uint *)&head_sbs_g[ihead][size_id];
	if(try_lock(&head_locks_g[ihead][size_id])) {
		// locked successfully, check if really need replacing blocks
		uint new_head = SB_NONE;
		//uint roomy_threshold = size_infos_g[size_id].roomy_threshold;
		//uint sparse_threshold = size_infos_g[size_id].sparse_threshold;
		//if(uc_cur_head == cur_head) 
		if(cur_head == SB_NONE || 
			 sb_count(*(volatile uint *)&sb_counters_g[cur_head]) >=
			 size_infos_g[size_id].roomy_threshold) {
			
#if CACHE_HEAD_SBS
			new_head = cached_sbs_g[ihead][size_id];
			cached_sbs_g[ihead][size_id] = SB_NONE;
#endif
			// this can happen, e.g., on start
			if(new_head == SB_NONE)
				new_head = find_sb_for_size(size_id, chunk_id);
			//new_head = chead;
			//assert(new_head != SB_NONE);
			if(new_head != SB_NONE) {
				// set the new head, so that allocations can continue
				// TODO: check if this threadfence is necessary
				//__threadfence();
				// if(new_head != SB_NONE) {
				//  	printf("new head = %d with count = %d\n", new_head, 
				//  				 (uint)sb_count(sb_counters_g[new_head]));
				// }
				__threadfence();
				*(volatile uint *)&head_sbs_g[ihead][size_id] = new_head;
				__threadfence();
				// detach current head
				if(cur_head != SB_NONE)
					detach_head(cur_head);
#if CACHE_HEAD_SBS
				cached_sbs_g[ihead][size_id] = find_sb_for_size(size_id, chunk_id);
#endif
				__threadfence();
			}  // if(found new head)
		} else {
			// looks like we read stale data at some point, just re-read head
			new_head = *(volatile uint *)&head_sbs_g[ihead][size_id];
		}
		unlock(&head_locks_g[ihead][size_id]);
		//__threadfence();
		//uint64 t2 = clock64();
		//printf("needed %lld cycles to find new head slabs\n", t2 - t1);
		return new_head;
	} else {
		// someone else working on current head superblock; 
		while(true) {
			if(*(volatile uint *)&head_sbs_g[ihead][size_id] != cur_head ||
				 *(volatile uint *)&head_locks_g[ihead][size_id] == 0)
				//if(*(volatile uint *)&head_locks_g[ihead][size_id] == 0);
				break;
		}
		return *(volatile uint *)&head_sbs_g[ihead][size_id];
	}
}  // new_sb_for_size

	/** allocates memory inside the superblock 
			@param isb the superblock inside which to allocate
			@param [in,out] iblock the block from which to start searching
			@param size_id the size id for the allocation
			@returns the pointer to the allocated memory, or 0 if unable to allocate
	*/
__device__ __forceinline__ void *sb_alloc_in
(uint ihead, uint isb, uint &ichunk, uint size_id, bool &needs_new_head) {
	size_info_t *size_info = &size_infos_g[size_id];
	if(isb == SB_NONE) {
		needs_new_head = true;
		return 0;
	}
	//void *p = 0;
	void *p = 0;
	uint *block_bits = sb_block_bits(isb);
	//superblock_t sb = sbs_g[isb];
	void *sbptr = sbs_g[isb].ptr;
	uint nchunks = size_infos_g[size_id].nchunks_in_block;

	uint iword, ibit, old_word;
	bool reserved = false;
	// iterate until successfully reserved
	for(uint itry = 0; itry < MAX_NTRIES; itry++) {
		// try reserve
		iword = ichunk / WORD_SZ;
		ibit = ichunk % WORD_SZ;
		uint alloc_mask = ((1 << nchunks) - 1) << ibit;
		old_word = atomicOr(block_bits + iword, alloc_mask);
		if(!(old_word & alloc_mask)) {
			// initial reservation successful
			reserved = true;
			break;
		} else {
			if(~old_word & alloc_mask) {
				// memory was partially allocated, need to roll back
				atomicAnd(block_bits + iword, ~alloc_mask | (old_word & alloc_mask));
			}
			if(itry % CHECK_NTRIES == CHECK_NTRIES - 1) {
				// check the counter
				uint count = sb_count(*(volatile uint *)&sb_counters_g[isb]);
				if(count >= size_info->busy_threshold)
					break;
			}
			ichunk = (ichunk + size_info->hash_step) % size_info->nchunks;
			//ichunk = (ichunk + size_info->hash_step) & (size_info->nchunks - 1);
		}
	}
	if(reserved) {
		// increment counter
		uint inc_mask = sb_ctr_inc
			(size_id, size_info->chunk_id, isb, nchunks);
		//if(!sb_ctr_inc(size_id, isb, 1)) {
		if(!(inc_mask & 1)) {
			// reservation unsuccessful (slab was freed), cancel it
			sb_counter_dec(&sb_counters_g[isb], nchunks);
			//atomicAnd(block_bits + iword, ~(1 << ibit));
			uint alloc_mask = ((1 << nchunks) - 1) << ibit;
			atomicAnd(block_bits + iword, ~alloc_mask | (old_word & alloc_mask));
			reserved = false;
			needs_new_head = true;
		} else {
			if(inc_mask & 2)
				needs_new_head = true;
			p = (char *)sbptr + ichunk * size_info->chunk_sz;
			// write allocation size
			// TODO: support chunks of other size
			uint *alloc_sizes = sb_alloc_sizes(isb);
			sb_set_alloc_size(alloc_sizes, ichunk, nchunks);
		}
	} else
		needs_new_head = true;
	return p;
}  // sb_alloc_in
