# Halloc GPU memory allocator, version 0.11

# Intro #

Halloc is a high-throughput malloc/free-style dynamic memory allocator for
NVidia GPUs. It is based on using bit arrays to represent free blocks and
using a hash function to quickly search for free blocks. This idea, combined
with clever slab management and performance tuning, enables a really fast
allocator. Halloc achieves more than 1.5 bln. mallocs/s (more than 1
bln. malloc/free pairs/s) on K20X on 16-byte allocations, with tens of
thousands of GPU threads and more than 100 MiB allocated. This is much higher
than other state-of-the-art GPU allocators. In addition, Halloc's performance is
also more stable. This makes halloc suitable for use in GPGPU applications
requiring fast dynamic memory management. Halloc is mainly designed for small
allocation sizes, and delegates allocations larger than 3KiB to CUDA allocator.


# Requirements #

Software: CUDA 5.0 or higher (tested with 6.5)
Hardware: Compute Capability 2.0 or higher (tested on CC 3.5 devices K20X and
K40).


# Compiling #

To compile halloc library, type (in project's top directory):

    make

To run correctness tests (**CAUTION: takes a lot of time!**):

    make test

To build correctness tests without running them:

    make build-corr

To build performance tests without running them:
	 
    make build-perf

Performance tests are then located in `./tst/perf/bin` directory, and can be
invoked individually, e.g.

    ./tst/perf/bin/throughput
    ./tst/perf/bin/phase-throughput -f0.95 -F0.05 -e0.91 -g5 -t128

To install, edit `PREFIX` variable in the `makefile` to your desired install
directory (default is `$(HOME)/usr`) and type:

    make install

To uninstall:

	make uninstall


# Using Halloc #

See `samples/` directory for samples using Halloc.

## Compiling Your Program ##

The GPU application needs to be compiled with Halloc static library using
separate device compilation and linking. Assuming that the variable `PREFIX`
contains the installation prefix, and `myprog.cu` is the file being compiled, this
can be done as follows:

    nvcc -arch=sm_35 -O3 -I $(PREFIX)/include -dc myprog.cu -o myprog.o
    nvcc -arch=sm_35 -O3 -L $(PREFIX)/lib -lhalloc -o myprog myprog.o


## Halloc API ##

The functions defined by Halloc are in the `halloc.h` file, which needs to be
included into your code to use Halloc:

    #include <halloc.h>

Before using Halloc functions on device, it has to be initialized with `ha_init()`
function:

    void ha_init(halloc_opts_t opts = halloc_opts_t());

It can be given a full `halloc_opts_t` structure to control fine halloc
parameters, such as slab size or fraction of used chunks at which the slab is
considered "busy". It can also be called just with specifying amount of memory
to allocate (in bytes), or completely without any parameter list to keep defaults:

    ha_init(512 * 1024 * 1024);  // pass memory to allocate
    ha_init();  // use default amount of memory

Halloc defines two functions, `hamalloc()` to allocate and `hafree()` to free memory
(`malloc()` and `free()` are used by CUDA allocator, therefore Halloc has to use other
names). These functions can only be called from device code.

    void *hamalloc(size_t nbytes);
	
    void hafree(void *p);

Otherwise, these functions have pretty much the same behavior as standard C
`malloc()`/`free()`, and can be used in a similar way.

Allocating/freeing an array:

    int *p = (int *)hamalloc(8 * sizeof(int));
    p[0] = 0;
    p[1] = threadIdx.x;
    p[2] = 2;
    // ...
    // free the array
    hafree(p);


Allocating a list:

    // allocate a list
    typedef struct list_ {
	  int element;
	  struct list_ *next;
    } list;
    
    // ...
	
    list *l = (list *)hamalloc(sizeof(list));
    l->element = 1;
    l->next = (list *)hamalloc(sizeof(list));
    l->next->element = 2;
    l->next->next = NULL;


`hamalloc()` accepts the number of bytes to allocate, and returns the
pointer to allocated memory, or `NULL` if memory cannot be allocated. Similarly,
`hafree()` accepts either a pointer returned by `hamalloc()` or `NULL`, and
frees the memory previously allocated. Naturally, `hamalloc()` and `hafree()`
are thread-safe, and can be called simultaneously by threads of the same or
different kernels. `hamalloc()` allocations persist across kernel invocations,
and can be used in other kernel calls. Pointers allocated by `hamalloc()` can
only be freed by `hafree()`; they cannot be deallocated, e.g., by calls to
`cudaFree()`/`free()` on host or device.

`ha_shutdown()` is intended to free resources used by Halloc, but is currently a
no-op.


## Limitations ##

* There is currently no way to change parameters or allocate more memory after
Halloc has been initialized.
* `ha_shutdown()` does not free any resources, and is a no-op.


# Bugs #

Though correctness tests pass successfully, this provies nothing, of
course. Some bugs are most likely there ;)
