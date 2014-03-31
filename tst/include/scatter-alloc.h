#ifndef SCATTER_ALLOC_H_
#define SCATTER_ALLOC_H_

extern "C" {
	void* sc_init_heap(size_t memsize = 8*1024U*1024U);
	__device__ void *scmalloc(uint nbytes);
	__device__ void scfree(void *p);
}

#endif

