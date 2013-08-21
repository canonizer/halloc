#ifndef SCATTER_ALLOC_WRAPPER_H_
#define SCATTER_ALLOC_WRAPPER_H_

/** @file scatter-alloc-wrapper.h wrapper class for ScatterAlloc allocator */

#include "common.h"
#include <scatter-alloc.h>

class ScatterAlloc {
public:
	static void init(const CommonOpts &opts) {
		sc_init_heap(opts.memory);
	}

	static inline __device__ void *malloc(uint nbytes) {
		return scmalloc(nbytes);
	}

	static inline __device__ void free(void *p) {
		scfree(p);
	}

	static void shutdown(void) {
	}

}; 

#endif
