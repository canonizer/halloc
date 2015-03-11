/** @file size-infos.cuh implementation of some stuff related to size
		information */

__device__ __forceinline__ const size_info_t *info_for_size(uint size_id) {
#if __CUDA_CC__ >= 500
	// on Maxwell, use the __constant__ array
	return &size_infos_cg[size_id];
#else
	// below Maxwell, use the array in __device__ memory
	return &size_infos_dg[size_id];
#endif
}  // info_for_size
