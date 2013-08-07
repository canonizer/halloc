/** @file forward-globals.cuh global variables that are used in many .cuh files,
		and thus require a forward declaration. This file is included into halloc.cu
		before all other .cuh files */

/** real possible number of superblocks (based on device memory and superblock
		size) */
__constant__ uint nsbs_g;

/** superblock size (common for all superblocks, power-of-two) */
__constant__ uint sb_sz_g;
/** superblock size shift (for fast division operations) */
__constant__ uint sb_sz_sh_g;
