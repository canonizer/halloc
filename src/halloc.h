#ifndef HALLOC_H_
#define HALLOC_H_

/** @file hamalloc.h header for halloc allocator */
#ifdef HALLOCLIB_COMPILING
#define HALLOC_EXTERN 
#else
#define HALLOC_EXTERN extern
#endif

extern "C" {

/** memory allocation */
HALLOC_EXTERN __device__ void *hamalloc(uint nbytes);

/** freeing the memory */
HALLOC_EXTERN __device__ void hafree(void *p);

/** initializes memory allocator host-side
		@param memory amount of memory which should be made available for allocation
 */
void ha_init(size_t memory);

/** shuts down memory allocator host-side */
void ha_shutdown(void);

}

#endif
