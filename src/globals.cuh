/** @file forward-globals.cuh global variables that are used in many .cuh files,
		and thus require a forward declaration. This file is included into halloc.cu
		before all other .cuh files */

/** real possible number of superblocks (based on device memory and superblock
		size) */
static __constant__ uint nsbs_g;

/** superblock size (common for all superblocks, power-of-two) */
static __constant__ uint sb_sz_g;
/** superblock size shift (for fast division operations) */
static __constant__ uint sb_sz_sh_g;

/** real number of sizes */
static __constant__ uint nsizes_g;

/** slab descriptors */
static __device__ superblock_t sbs_g[MAX_NSBS];
/** slab pointers (stored separately from descriptors, as they do not change) */
__attribute__((aligned(128))) static __device__ void *sb_ptrs_g[MAX_NSBS];
/** slab (non-distributed) counters */
static __device__ uint sb_counters_g[MAX_NSBS];
