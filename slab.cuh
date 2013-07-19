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

/** the set of all unallocated superblocks */
__device__ sbset_t unallocated_sbs_g;
/** the set of all free superblocks */
__device__ sbset_t free_sbs_g;
/** the set of "roomy" superblocks for the size */
__device__ sbset_t roomy_sbs_g[MAX_NSIZES];

/** gets block bits for superblock */
__device__ inline uint *sb_block_bits(uint sb) {
	return block_bits_g + sb * nsb_bit_words_g;
}  // sb_block_bits

/** gets the alloc sizes for the superblock */
__device__ inline uint *sb_alloc_sizes(uint sb) {
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
	//((unsigned char *)alloc_words)[ichunk] = size_id;
	//__threadfence();
}  // sb_set_alloc_size

/** gets (and resets) allocation size for the allocation */
__device__ inline uint sb_get_reset_alloc_size(uint *alloc_words, uint ichunk) {
	// unsigned char *alloc_bytes = (unsigned char *)alloc_words;
	// return alloc_bytes[ichunk];
	uint iword = ichunk / 4, ibyte = ichunk % 4, shift = ibyte * 8;
	uint mask = 0xffu << shift;
	return (atomicOr(&alloc_words[iword], mask) >> shift) & 0xffu;
}  // sb_get_reset_alloc_size

/** increment the non-distributed counter of the superblock 
		size_id is just ignored, allocation size is expressed in chunks
		@returns old counter value
 */
__device__ __forceinline__ void sb_ctr_inc
(uint size_id, uint sb_id, uint alloc_sz) {
	bool want_inc = true;
	uint mask, lid = threadIdx.x % WARP_SZ;
	while(mask = __ballot(want_inc)) {
		uint leader_lid = __ffs(mask) - 1, leader_sb_id = sb_id;
		leader_sb_id = __shfl((int)leader_sb_id, leader_lid);
		// allocation size is same for all superblocks
		uint change = alloc_sz * __popc(__ballot(sb_id == leader_sb_id));
		if(lid == leader_lid)
			atomicAdd(&sb_counters_g[sb_id], change);
		want_inc = want_inc && sb_id != leader_sb_id;
	}  // while
}  // sb_ctr_inc

/** increment the non-distributed counter of the superblock 
		size_id is just ignored, allocation size is expressed in chunks
 */
__device__ __forceinline__ void sb_ctr_dec
(uint size_id, uint sb_id, uint alloc_sz) {
	bool want_inc = true;
	uint mask, lid = threadIdx.x % WARP_SZ;
	while(mask = __ballot(want_inc)) {
		uint leader_lid = __ffs(mask) - 1, leader_sb_id = sb_id;
		leader_sb_id = __shfl((int)leader_sb_id, leader_lid);
		// allocation size is same for all superblocks
		uint change = alloc_sz * __popc(__ballot(sb_id == leader_sb_id));
		if(lid == leader_lid) {
			uint old_val = atomicSub(&sb_counters_g[sb_id], change);
			uint new_val = old_val - change;
			if((old_val + SB_FREE_STEP - 1) % SB_FREE_STEP != 
				 (new_val + SB_FREE_STEP - 1) % SB_FREE_STEP) {
				uint threshold = size_infos_g[size_id].roomy_threshold;
				if(new_val <= threshold) {
					// mark superblock as roomy for current size
					sbset_add_to(&roomy_sbs_g[size_id], sb_id);
				}
			}
		}
		want_inc = want_inc && sb_id != leader_sb_id;	
	}  // while
}  // sb_ctr_dec

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
	if(sb_counters_g[isb] >= size_info.busy_threshold)
		return 0;
	uint iword;
	bool reserved = false;
	// iterate until successfully reserved
	for(uint itry = 0; itry < MAX_NTRIES; itry++) {
		//for(uint i = 0; i < 1; i++) {
		// try reserve
		iword = iblock / WORD_SZ;
		uint ibit = iblock % WORD_SZ, alloc_mask = 1 << ibit;
		uint old_word = atomicOr(block_bits + iword, alloc_mask);
		if(!(old_word & alloc_mask)) {
			// reservation successful
			reserved = true;
			break;
		} else {
			iblock = (iblock + size_info.hash_step) % size_info.nblocks;
			//iblock = (iblock + size_info.hash_step) &	(size_info.nblocks - 1);
		}
	}
	if(reserved) {
		// write allocation size
		p = (char *)sb.ptr + iblock * size_info.block_sz;
		// TODO: support chunks of other size
		//uint *alloc_sizes = sb_alloc_sizes(isb);
		//sb_set_alloc_size(alloc_sizes, iblock, size_id);
		//sb_ctr_inc(~0, isb, size_info.block_sz / BLOCK_STEP);
		sb_ctr_inc(~0, isb, 1);
	}
	return p;
}  // sb_alloc_in
