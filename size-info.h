#ifndef HALLOC_SIZE_INFO_H_
#define HALLOC_SIZE_INFO_H_

/** @file size-info.h information and definitions related to sizes */

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

/** maximum number of sizes supported */
#define MAX_NSIZES 64
/** a "no-size" constant */
#define SZ_NONE (~0)
/** block step (16 bytes by default), a power of two */
#define BLOCK_STEP 16
/** minimum block size (a power of two) */
#define MIN_BLOCK_SZ 16
/** maximum block size (a power of two) */
#define MAX_BLOCK_SZ 256

#endif
