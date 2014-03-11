#ifndef SCATTER_ALLOC_WRAPPER_H_
#define SCATTER_ALLOC_WRAPPER_H_

/** @file scatter-alloc-wrapper.h wrapper class for ScatterAlloc allocator */
#ifdef WITH_SCATTER

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

	static double extfrag(bool ideal) {
		return 0;
	}

	static void shutdown(void) {
	}

}; 

#endif

#endif
