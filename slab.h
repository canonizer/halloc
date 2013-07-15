#ifndef HALLOC_SLAB_H_
#define HALLOC_SLAB_H_

/** @file slab.h slab (superblock) header file */
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

/** number of superblock counters */
#define NSB_COUNTERS 16
/** a step at which to  change the value of a distributed counter */
#define SB_DISTR_STEP 8
/** a step to change the value of main counter */
#define SB_MAIN_STEP 512
/** maximum number of tries inside a superblock after which the allocation
		attempt is abandoned */
#define MAX_NTRIES 512
/** a "no-sb" constant */
#define SB_NONE (~0)
/** default superblock size, in bytes */
#define SB_SZ_SH (10 + 10 + 3)
#define SB_SZ (1 << SB_SZ_SH)

#endif
