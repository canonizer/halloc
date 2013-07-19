/** @file sbset.cuh slab set implementation */

extern __constant__ uint nsbs_g;

__device__ inline uint sbset_get_from(sbset_t *sbset, uint start) {
	if(start == SB_NONE)
		start = 0;
	// TODO: maybe do several trials to be sure
	uint start_iword = start / WORD_SZ, start_ibit = start % WORD_SZ;
	for(uint iword = start_iword; iword < nsbs_g / WORD_SZ; iword++)
		for(uint word = (*sbset)[iword], ibit = __ffs(word) - 1; word;
				word &= ~(1 <<	ibit), ibit = __ffs(word) - 1) {
			if(iword > start_iword || ibit > start_ibit) {
				// try locking the bit
				uint mask = 1 << ibit;
				if(atomicAnd(&(*sbset)[iword], ~mask) & mask)
					return iword * WORD_SZ + ibit;
			}  // if(current bit can be locked)
		}
	return SB_NONE;
}  // sbset_get_from
