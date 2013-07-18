#ifndef HALLOC_SLAB_H_
#define HALLOC_SLAB_H_

/** @file slab.h slab (superblock) header file */
/** superblock descriptor type; information is mostly changing; note that during
		allocation, a superblock is mostly identified by superblock id */
typedef struct {
	/** superblock size id */
	uint size_id;
	/** pointer to memory owned by superblock */
	void *ptr;
} superblock_t;

/** a step to check whether the slab can be moved to another free category */
#define SB_FREE_STEP 2048
/** maximum number of tries inside a superblock after which the allocation
		attempt is abandoned */
#define MAX_NTRIES 256
/** a "no-sb" constant */
#define SB_NONE (~0)
/** default superblock size, in bytes */
#define SB_SZ_SH (10 + 10 + 2)
#define SB_SZ (1 << SB_SZ_SH)

#endif
