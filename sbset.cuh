/** @file sbset.cuh slab set implementation */

extern __constant__ uint nsbs_g;

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
