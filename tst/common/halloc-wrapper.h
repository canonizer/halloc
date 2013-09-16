#ifndef HALLOC_WRAPPER_H_
#define HALLOC_WRAPPER_H_

/** @file halloc-wrapper.h wrapper class for halloc allocator */

#include "common.h"
#include <halloc.h>

class Halloc {
public:
	static void init(const CommonOpts &opts) {
		halloc_opts_t halloc_opts(opts.memory);
		halloc_opts.halloc_fraction = opts.halloc_fraction;
		halloc_opts.busy_fraction = opts.busy_fraction;
		halloc_opts.roomy_fraction = opts.roomy_fraction;
		halloc_opts.sparse_fraction = opts.sparse_fraction;
		halloc_opts.sb_sz_sh = opts.sb_sz_sh;
		ha_init(halloc_opts);
	}

	static inline __device__ void *malloc(uint nbytes) {
		return hamalloc(nbytes);
	}

	static inline __device__ void free(void *p) {
		hafree(p);
	}

	static double extfrag(bool ideal) {
		return ha_extfrag(ideal);
	}

	static void shutdown(void) {
		ha_shutdown();
	}

}; 

#endif
