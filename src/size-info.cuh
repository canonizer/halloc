/** @file size-infos.cuh implementation of some stuff related to size
		information */

// TODO: with __constant__, better for hafree(), with __device__, better for
// hamalloc(); hence probably need 2 arrays, once for use in each of the functions
/** information on sizes */
__device__ size_info_t size_infos_g[MAX_NSIZES];

/** same data, but in different memory */
// __device__ size_info_t size_infos_dg[MAX_NSIZES];
