#ifndef HALLOC_H_
#define HALLOC_H_

/** @file hamalloc.h header for halloc allocator */

/** memory allocation */
__device__ void *hamalloc(uint nbytes);

/** freeing the memory */
__device__ void hafree(void *p);

/** initializes memory allocator host-side */
void ha_init(void);

/** shuts down memory allocator host-side */
void ha_shutdown(void);

#endif
