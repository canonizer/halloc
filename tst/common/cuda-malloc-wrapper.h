#ifndef CUDA_MALLOC_WRAPPER_H_
#define CUDA_MALLOC_WRAPPER_H_

/** @file cuda-malloc-wrapper.h wrapper class for CUDA malloc allocator */

#include "common.h"

class CudaMalloc {
public:
	static void init(const CommonOpts &opts) {
		cucheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, opts.memory));
	}

	static inline __device__ void *malloc(uint nbytes) {
		return ::malloc(nbytes);
	}

	static inline __device__ void free(void *p) {
		::free(p);
	}

	static double extfrag(bool ideal) {
		return 0;
	}

	static void shutdown(void) {}

}; 

#endif
