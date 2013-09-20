/** @file sbset.cuh slab set implementation */

//extern __constant__ uint nsbs_g;

__device__ inline uint sbset_get_from(sbset_t sbset) {
	// the condition is always true, but the compiler doesn't know that
	// without it, performance somewhat drops
#if SBSET_CTR
	if(nsbs_g) {
		int old = *(volatile int*)&sbset[SB_SET_SZ - 1];
		//int old = atomicAdd((int *)&sbset[SB_SET_SZ - 1], 0);
		if(old <= 0)
			return SB_NONE;
	}
#endif
	// then get it
	for(uint iword = 0; iword < nsbs_g / WORD_SZ; iword++) {
		// atomicOr() also works good here
		uint word = *(volatile uint *)&sbset[iword];
		//uint word = atomicOr(&sbset[iword], 0);
		while(word) {
			uint ibit = __ffs(word) - 1;
			// try locking the bit
			uint mask = 1 << ibit;
			if(atomicAnd(&sbset[iword], ~mask) & mask) {
#if SBSET_CTR
				atomicSub(&sbset[SB_SET_SZ - 1], 1);
#endif
				return iword * WORD_SZ + ibit;
			}
			word &= ~mask;
		}
	}
	return SB_NONE;
}  // sbset_get_from
