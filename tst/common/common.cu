/** @file common.cu implementation of common library for Halloc testing */

#define COMMONTEST_COMPILING

#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/sort.h>

#include "common.h"

using namespace thrust;

// parsing options
const char *opts_usage_g = 
	"usage: <test-name> <options>\n"
	"\n"
	"supported options are to be added later\n";

void print_usage_and_exit(int exit_code) {
	printf("%s", opts_usage_g);
	exit(exit_code);
}  // print_usage_and_exit

double parse_double(char *str, double a = 0.0, double b = 1.0) {
	double r;
	if(sscanf(str, "%lf", &r) != 1) {
		fprintf(stderr, "%s is not a double value\n", str);
		print_usage_and_exit(-1);
	}
	if(r < a || r > b) {
		fprintf(stderr, "double value %lf is not in range [%lf, %lf]\n", r, a, b);
		print_usage_and_exit(-1);
	}
	return r;
}  // parse_double

int parse_int(char *str, int a = INT_MIN, int b = INT_MAX) {
	int r;
	if(sscanf(str, "%d", &r) != 1) {
		fprintf(stderr, "%s is not an integer value or too big\n", &r);
		print_usage_and_exit(-1);
	}
	if(r < a || r > b) {
		fprintf(stderr, "integer value %d is not in range [%d, %d]\n", r, a, b);
		print_usage_and_exit(-1);
	}
	return r;
}  // parse_int

char *allocator_types[] = {
	"cuda", "halloc", "scatter", "xmalloc"
};

char *distr_types[] = {
	"uniform", "expuniform", "expequal"
};

static uint parse_enum(char *str, char *name, char **vals, uint top) {
	int istr;
	for(istr = 0; istr < top - 1; istr++)
		if(!strcmp(str, vals[istr]))
			break;
	istr++;
	if(istr == top) {
		printf("%s: invalid %s name\n", str, name);
		print_usage_and_exit(-1);
	}
	return istr;
}  // parse_enum

AllocatorType parse_allocator(char *str) {
	return (AllocatorType)parse_enum
		(str, "allocator", allocator_types, AllocatorTopNone);
}  // parse_allocator

DistrType parse_distr(char *str) {
	return (DistrType)parse_enum
		(str, "distribution", distr_types, DistrTopNone);
}  // parse_distr

void CommonOpts::parse_cmdline(int argc, char **argv) {
	static const char *common_opts_str = ":ha:m:C:B:R:D:b:n:t:T:s:S:l:i:f:q:g:d:p:P:";
	int c;
	int period_sh, ndevices;
	cucheck(cudaGetDeviceCount(&ndevices));
	bool nthreads_explicit = false, min_alloc_explicit = false, 
		max_alloc_explicit = false;
	while((c = getopt(argc, argv, common_opts_str)) != -1) {
		switch(c) {
			// general options (and errors)
		case 'h':
			print_usage_and_exit(0);
			break;
		case ':':
			fprintf(stderr, "missing argument for option %c\n", optopt);
			print_usage_and_exit(-1);
			break;
		case '?':
			fprintf(stderr, "unknown option -%c\n", optopt);
			print_usage_and_exit(-1);
			break;

			// allocator options
		case 'a':
			allocator = parse_allocator(optarg);
			break;
		case 'm':
			memory = parse_int(optarg, 4096);
			break;
		case 'C':
			halloc_fraction = parse_double(optarg);
			break;
		case 'B':
			busy_fraction = parse_double(optarg);
			break;
		case 'R':
			roomy_fraction = parse_double(optarg);
			break;
		case 'b':
			sb_sz_sh = parse_int(optarg, 20, 26);
			break;

			// test options
		case 'D':
			device = parse_int(optarg, 0, ndevices - 1);
			break;
		case 'n':
			nthreads = parse_int(optarg, 0);
			nthreads_explicit = true;
			break;
		case 't':
			ntries = parse_int(optarg, 1);
			break;
		case 'T':
			bs = parse_int(optarg, 1, 1024);
			break;
		case 's':
			min_alloc_explicit = true;
			alloc_sz = parse_int(optarg, 0);
			if(max_alloc_explicit) { 
				if(max_alloc_sz < alloc_sz) {
					fprintf(stderr, "max allocation size should be >= " 
									"min allocation	size\n");
					print_usage_and_exit(-1);
				}
			} else
				max_alloc_sz = alloc_sz;
			break;
		case 'S':
			max_alloc_explicit = true;
			//printf("before setting max_alloc_sz = %d\n", max_alloc_sz);
			max_alloc_sz = parse_int(optarg, 0);
			//printf("after setting max_alloc_sz = %d\n", max_alloc_sz);
			if(min_alloc_explicit) {
				if(max_alloc_sz < alloc_sz) {
					fprintf(stderr, "max allocation size should be >= " 
									"min allocation	size\n");
					print_usage_and_exit(-1);
				}
			} else
				alloc_sz = max_alloc_sz;
			break;
		case 'l':
			nallocs = parse_int(optarg, 1);
			break;
		case 'i':
			niters = parse_int(optarg, 1);
			break;
		case 'f':
			alloc_fraction = parse_double(optarg);
			break;
		case 'q':
			period_sh = parse_int(optarg, 0, 31);
			period_mask = period_sh > 0 ? ((1 << period_sh) - 1) : 0;
			break;
		case 'g':
			group_sh = parse_int(optarg, 0, 31);
			break;
		case 'd':
			distr_type = parse_distr(optarg);
			break;
		case 'p':
			palloc = (float)parse_double(optarg);
			break;
		case 'P':
			pfree = (float)parse_double(optarg);
			break;

		default:
			fprintf(stderr, "this simply should not happen when parsing options\n");
			print_usage_and_exit(-1);
			break;
		}  // switch
	}

	// cap memory to fraction of device memory
	int device;
	cucheck(cudaGetDevice(&device));
	cudaDeviceProp props;
	cucheck(cudaGetDeviceProperties(&props, device));
	size_t dev_memory = props.totalGlobalMem;
	memory = min((unsigned long long)memory, 
							 (unsigned long long)(0.75 * dev_memory));

	// cap number of threads for CUDA allocator
	if(allocator == AllocatorCuda && !nthreads_explicit)
		nthreads = min(nthreads, 32 * 1024);
	// check probabilities
	// if(palloc + pfree > 1) {
	// 	printf("palloc = %lf, pfree = %lf, total > 1\n", (double)palloc, 
	// 				 (double)pfree);
	// 	print_usage_and_exit(-1);
	// }

	// recompute some fields
	recompute_fields();
	//printf("min_sz = %d, max_sz = %d\n", alloc_sz, max_alloc_sz);
}  // parse_cmdline

double CommonOpts::expected_sz(void) {
	if(alloc_sz == max_alloc_sz)
		return alloc_sz;
	switch(distr_type) {
	case DistrUniform:
		return ((double)alloc_sz + max_alloc_sz) / 2;
	case DistrExpUniform:
		{
			double expectation = 0;
			for(uint sh = 0; sh <= max_alloc_sh; sh++) {
				double lo = alloc_sz << sh;
				double hi = min((alloc_sz << (sh + 1)) - 1, max_alloc_sz);
				expectation += (lo + hi) / 2;
			}
			expectation /= max_alloc_sh + 1;
			return expectation;
		}
	case DistrExpEqual:
		{
			double expectation = 0, probab = 1;
			for(uint sh = 0; sh <= max_alloc_sh; sh++) {
				if(sh < max_alloc_sz)
					probab /= 2;
				double lo = alloc_sz << sh;
				double hi = min((alloc_sz << (sh + 1)) - 1, max_alloc_sz);
				expectation += (lo + hi) / 2 * probab;
			}
			//expectation /= max_alloc_sh + 1;
			return expectation;
		}
	default:
		// this shouldn't happen
		fprintf(stderr, "invalid distribution type\n");
		exit(-1);
	}  // switch
}

double CommonOpts::total_nallocs(void) {
	return (double)nptrs_cont(nthreads) * nallocs * niters * ntries;
}

double CommonOpts::total_sz(void) {
	return expected_sz() * total_nallocs();
}

void CommonOpts::recompute_fields(void) {
	// recompute max_alloc_sh
	max_alloc_sh = 0;
	while(max_alloc_sz >= alloc_sz << (max_alloc_sh + 1))
		max_alloc_sh++;
}  // recompute_fields

void drandom_init(const CommonOpts &opts) {
	srandom((uint)time(0));

	// TODO: somehow standardize this number
	const uint MAX_NTHREADS = 8 * 1024 * 1024;
	uint n = max(MAX_NTHREADS, opts.nthreads);
	size_t sz = n * sizeof(uint);
	uint *d_random_states, *h_random_states;

	// allocate memory
	cucheck(cudaMalloc((void **)&d_random_states, sz));
	h_random_states = (uint *)malloc(sz);

	// initialize random values, respect groups
	uint gp = opts.group() * opts.period();
	uint seed;
	for(uint i = 0; i < n; i++) {
		if(i % gp == 0)
			seed = random();
		h_random_states[i] = seed;
	}
	cucheck(cudaMemcpy(d_random_states, h_random_states, sz, 
										 cudaMemcpyHostToDevice));
	free(h_random_states);
	
	// initialize device variable
	cuset(random_states_g, uint *, d_random_states);	
}  // drandom_init

void drandom_shutdown(const CommonOpts &opts) {
	// currently nothing is done
}

struct ptr_is_nz {
	void **ptrs;
	uint *ctrs;
	CommonOpts opts;
	__host__ __device__ ptr_is_nz
	(void **ptrs, uint *ctrs, const CommonOpts &opts) 
		: opts(opts), ptrs(ptrs), ctrs(ctrs) {}
	__host__ __device__ bool operator()(int i) { 
		if(opts.is_thread_inactive(i)) 
			return true;
		else {
			uint ctr = ctrs ? ctrs[i] : 1;
			for(uint ialloc = 0; ialloc < ctr; ialloc++) {
				if(!ptrs[ialloc * opts.nthreads + i])
					return false;
			}
			return true;
		}
	}  // operator ()
};  // ptr_is_nz

bool check_nz(void **d_ptrs, uint *d_ctrs, uint nptrs, const CommonOpts &opts) {
	return all_of
		(counting_iterator<int>(0), counting_iterator<int>(nptrs),
		 ptr_is_nz(d_ptrs, d_ctrs, opts));
}  // check_nz

__global__ void copy_cont_k
(void **to, void **from, uint *ctrs, uint *fill_ctr, CommonOpts opts) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if(opts.is_thread_inactive(i))
		return;
	uint nallocs = ctrs ? ctrs[i] : opts.nallocs;
	uint pos = atomicAdd(fill_ctr, nallocs);
	for(uint ialloc = 0; ialloc < nallocs; ialloc++)
		to[pos + ialloc] = from[ialloc * opts.nthreads + i];
}  // copy_cont_k

/** a helper functor to check whether each pointer has enough room */
struct has_enough_room {
	uint64 *d_ptrs;
	size_t alloc_sz;
	int nptrs;
	__host__ __device__ has_enough_room
	(uint64 *d_ptrs, size_t alloc_sz, int	nptrs) 
		: d_ptrs(d_ptrs), alloc_sz(alloc_sz), nptrs(nptrs) {}
	__host__ __device__ bool operator()(int i) {
		if(i == nptrs - 1)
			return true;
		return d_ptrs[i] + alloc_sz <= d_ptrs[i + 1];
	}
};  // has_enough_room

/** a kernel which simply writes thread id at the address specified by each
		pointer in the passed array */
__global__ void write_tid_k(void **d_ptrs, int nptrs) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= nptrs)
		return;
	*(int *)d_ptrs[tid] = tid;
}  // write_tid_k

/** a helper functor to check tid written at each address */
struct check_tid {
	void **d_ptrs;
	__host__ __device__ check_tid(void **d_ptrs) : d_ptrs(d_ptrs) {}
	__host__ __device__ bool operator()(int tid) {
		return *(int *)d_ptrs[tid] == tid;
	}
};

bool check_alloc
(void **d_ptrs, uint *d_ctrs, uint nptrs, const CommonOpts &opts) {
	uint alloc_sz = opts.alloc_sz;
	//uint period = opts.period();
	if(!check_nz(d_ptrs, d_ctrs, nptrs, opts)) {
		fprintf(stderr, "cannot allocate enough memory\n");
		return false;
	}
	// first copy into a contiguous location
	void **d_ptrs_cont = 0;
	uint group = opts.group();
	int nptrs_cont;
	if(d_ctrs) {
		device_ptr<uint> dt_ctrs(d_ctrs);
		nptrs_cont = reduce(dt_ctrs, dt_ctrs + nptrs, 0, plus<int>());
	} else {
		nptrs_cont = opts.nptrs_cont(nptrs / opts.nallocs) * opts.nallocs;
	}
	cucheck(cudaMalloc((void **)&d_ptrs_cont, nptrs_cont * sizeof(void *)));

	uint *d_fill_ctr;
	cucheck(cudaMalloc((void **)&d_fill_ctr, sizeof(uint)));
	cucheck(cudaMemset(d_fill_ctr, 0, sizeof(uint)));
	uint bs = 128;
	copy_cont_k<<<divup(opts.nthreads, bs), bs>>>
		(d_ptrs_cont, d_ptrs, d_ctrs, d_fill_ctr, opts);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	
	// transform
	// 	(counting_iterator<int>(0), counting_iterator<int>(nptrs_cont),
	// 	 device_ptr<void *>(d_ptrs_cont), copy_cont(d_ptrs_cont, d_ptrs, opts));
	// sort the pointers
	device_ptr<uint64> dt_ptrs((uint64 *)d_ptrs_cont);
	sort(dt_ptrs, dt_ptrs + nptrs_cont);
	// check whether each pointer has enough room
	if(!all_of(counting_iterator<int>(0), counting_iterator<int>(nptrs_cont), 
						 has_enough_room((uint64 *)d_ptrs_cont, alloc_sz, nptrs_cont))) {
		fprintf(stderr, "allocated pointers do not have enough room\n");
		cucheck(cudaFree(d_ptrs_cont));
		return false;
	} 

	// do write-read test to ensure there are no segfaults
	//int bs = 128;
	write_tid_k<<<divup(nptrs_cont, bs), bs>>>(d_ptrs_cont, nptrs_cont);
	bool res = all_of(counting_iterator<int>(0), counting_iterator<int>(nptrs_cont), 
								check_tid(d_ptrs_cont));
	cucheck(cudaFree(d_ptrs_cont));
	return res;
}  // check_alloc
