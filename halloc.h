#ifndef HALLOC_H_
#define HALLOC_H_

/** @file hamalloc.h header for halloc allocator */

/** memory allocation; @nbytes is ignored */
__device__ void *hamalloc(size_t nbytes);

/** freeing the memory */
__device__ void hafree(void *p);

/** initializes memory allocator host-side */
void ha_init(void);

/** shuts down memory allocator host-side */
void ha_shutdown(void);

/** allocator parameters */
#define NBLOCKS (16 * 1024 * 1024)

#endif
