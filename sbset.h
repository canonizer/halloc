#ifndef HALLOC_SBSET_H_
#define HALLOC_SBSET_H_

/** @file sbset.h slab set definitions */

#include "utils.h"

/** superblock set type */
typedef uint sbset_t[SB_SET_SZ];

/** gets superblock from set (and removes it) */
__device__ inline uint sbset_get_from(sbset_t *sbset, uint start = 0);

/** adds ("returns") superblock to the set */
__device__ inline void sbset_add_to(sbset_t *sbset, uint sb) {
	uint iword = sb / WORD_SZ, ibit = sb % WORD_SZ;
	atomicOr(&(*sbset)[iword], 1 << ibit);
}  // sbset_add_to

/** removes the specified slab from set */
__device__ inline void sbset_remove_from(sbset_t *sbset, uint sb) {
	uint iword = sb / WORD_SZ, ibit = sb % WORD_SZ;
	atomicAnd(&(*sbset)[iword], ~(1 << ibit));
}

#endif
