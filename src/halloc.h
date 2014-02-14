#ifndef HALLOC_H_
#define HALLOC_H_

/** @file hamalloc.h header for halloc allocator */
#ifdef HALLOCLIB_COMPILING
#define HALLOC_EXTERN 
#else
#define HALLOC_EXTERN extern
#endif

//#define HALLOC_CPP

/** structure (class) for halloc allocator options */

struct halloc_opts_t {
	/** total amount of memory available for allocation, bytes */
	size_t memory;
	/** memory fraction available to halloc allocator, the rest goes to CUDA for
	larger allocations */
	double halloc_fraction;
	/** occupancy fraction at which a slab is considered busy */
	double busy_fraction;
	/** occupancy fraction at which a slab is considered roomy */
	double roomy_fraction;
	/** occupancy fraction at which a slab is considered sparse */
	double sparse_fraction;
	/** shift value for slab size (size in bytes) */
	int sb_sz_sh;
	/** default constructor which initializes the structure with default values */
	halloc_opts_t(size_t memory = 512 * 1024 * 1024) : 
		memory(memory), halloc_fraction(0.75), busy_fraction(0.835),
		roomy_fraction(0.6), sparse_fraction(0.012), sb_sz_sh(22)
	{}
};  // halloc_opts_t

/** memory allocation */
//HALLOC_EXTERN __device__ __noinline__ void *hamalloc(uint nbytes);
HALLOC_EXTERN __device__ __noinline__ void *hamalloc(size_t nbytes);

/** freeing the memory */
HALLOC_EXTERN __device__ __noinline__ void hafree(void *p);

/** initializes memory allocator host-side
		@param memory amount of memory which should be made available for allocation
 */
void ha_init(halloc_opts_t opts = halloc_opts_t());

/** shuts down memory allocator host-side */
void ha_shutdown(void);

/** gets current external fragmentation 
		@param ideal whether the ideal case is considered, i.e. all slabs do really
		get allocated from CUDA allocator memory; currently ignored and assumed false
 */
double ha_extfrag(bool ideal);

// overrides for malloc and free if requested; currently unstable
//#ifdef HALLOC_OVERRIDE_STDC
#if 0
__device__ void *malloc(uint nbytes) throw() { 
		return hamalloc(nbytes);
	}
inline __device__ void free(void *p) throw() { hafree(p); }
extern "C" __host__ void free(void *p) throw();
#endif

// overload new/delete C++ operators on device if requested
// currently doesn't make much sense: the compiler treats operator new very
// specially, and obviously links it against an external library, which kills
// all performance
#if defined(HALLOC_CPP)
#include <new>
//struct halloc_tag_t;
//typedef halloc_tag_t *halloc_t;
//#define halloc ((halloc_t)0)
__device__ void *operator new(size_t nbytes) throw(std::bad_alloc);
//__device__ void *operator new[](size_t nbytes) throw(std::bad_alloc);
__device__ void operator delete(void *p) throw();
//__device__ void operator delete[](void *p) throw();
#endif

#endif
