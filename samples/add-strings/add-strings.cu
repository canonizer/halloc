/** @file grid-points.cu a test where grid points are sorted into a grid */

#include <halloc.h>

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/** a macro for checking CUDA calls */
#define cucheck(call)																										\
	{																																			\
	cudaError_t cucheck_err = (call);																			\
	if(cucheck_err != cudaSuccess) {																			\
		const char* err_str = cudaGetErrorString(cucheck_err);							\
		fprintf(stderr, "%s (%d): %s in %s\n", __FILE__, __LINE__, err_str, #call);	\
		exit(-1);																														\
	}																																			\
	}

int divup(int a, int b) { return a / b + (a % b ? 1 : 0); }

typedef unsigned long long int uint64;

/** prefetches into L1 cache */
__device__ inline void prefetch_l1(const void *p) {
	asm("prefetch.global.L1 [%0];": :"l"(p));
}

/** prefetches into L2 cache */
__device__ inline void prefetch_l2(const void *p) {
	asm("prefetch.global.L2 [%0];": :"l"(p));
}

/** a random value in [a, b] range */
int random(int a, int b) {
	return a + random() % (b - a + 1);
	//return a;
}

/** an array filled with random values in [a, b] range, with contiguous groups
		of p values starting at p being the same */
void random_array(int *arr, size_t n, int p, int a, int b) {
	int v = 0;
	for(size_t i = 0; i < n; i++) {
		if(i % p == 0)
			v = random(a, b);
		arr[i] = v;
	}
}

void alloc_strs(char **strs, const int *lens, int n) {
#pragma omp parallel for
	for(int i = 0; i < n; i++) {
		int l = lens[i];
		char *str = (char *)malloc((l + 1) * sizeof(char));
		//strs[i] = (char *)malloc((l + 1) * sizeof(char));
		/*
		for(int j = 0; j < l; j++)
			str[j] = ' ';
			str[l] = 0; */
		strs[i] = str;
	}
}  //alloc_strs

/** a kernel that allocates and initializes an array of strings; memory for
		strings is allocated using halloc */
__global__ void alloc_strs_k
(char ** __restrict__ strs, 
 const int * __restrict__ lens, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= n)
		return;
	// allocate string (don't forget zero byte!)
	int l = lens[i];
	//if(i == 0)
	//	printf("l = %d\n", l);
	//if(i > n - 256)
	//	printf("i = %d, l = %d, n = %d\n", i, l, n);
	// char *str = (char *)hamalloc((l + 1) * sizeof(char));
	// for(int j = 0; j < l; j++)
	//  	str[j] = '0' + j;
	// str[l] = 0;

	uint64 *str = (uint64 *)hamalloc((l + 1) * sizeof(char));
	int l_i = (l + 1) / 8;
	for(int j = 0; j < l_i - 1; j++) {
	  str[j] = 0x2020202020202020ull;
	}
	str[l_i - 1] = 0x0020202020202020ull;

	// save string pointer
	strs[i] = (char *)str;
}  // alloc_strs_k

/** a kernel that frees memory allocated for strings */
__global__ void free_strs_k
(char ** __restrict__ strs, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= n)
		return;
	hafree(strs[i]);
}  // free_strs_k

void free_strs(char ** strs, int n) {
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
		free(strs[i]);
}  // free_strs

/** finds the zero byte in a long value, returns INT_MAX if not found */
__device__ inline int izero_byte(uint64 v) {
	int l = INT_MAX;
	#pragma unroll 8
	for(int i = 0; i < 8; i++) {
		if(((v >> i * 8) & 0xffu) == 0)
			l = min(l, i);
	}
	return l;
}  // zero_byte

// couple of helper device functions, analogous to C library
/** get the length of a string; it is assumed that s is at least 8-byte aligned */
__device__ inline int dstrlen(const char * __restrict__ s) {
	//int len = -1;
	int len = INT_MAX;
	//while(*s++) len++;
	//return len;
	const uint64 *s1 = (const uint64 *)s;
	int ll = 0;
	while(len == INT_MAX) {
	//while(true) {
		uint64 c1 = *s1++;
		#pragma unroll 8
		for(int i = 0; i < 8; i++) {
			//if(((c1 >> i * 8) & 0xffu) == 0)
			//	return len;
			//len++;
			if(((c1 >> i * 8) & 0xffu) == 0)
				len = min(len, ll + i);
		}
		ll++;
	}
	return len;
}  // strlen

/** concatenate two strings into the third string; all strings have been
		allocated, and the result has enough place to hold the arguments; 
		all pointers are assumed to be 8-byte aligned
*/
__device__ inline void dstrcat
(char *  __restrict__ c, const char * __restrict__ b, 
 const char * __restrict__ a) {
	// while(*c++ = *a++) {}
	// c--;
	// while(*c++ = *b++) {}
	uint64 *c1 = (uint64 *)c;
	const uint64 *a1 = (const uint64 *)a;
	const uint64 *b1 = (const uint64 *)b;
	uint64 cc = 0, aa = 0, bb = 0;
	int ccpos = 0;
	//uint cc1 = 0;
	// TODO: optimize computations for concatenation, similar to dstrlen()
	// copy first string
	int izb = INT_MAX;
	do {
		aa = *a1++;
		cc |= aa << ccpos;
		izb = izero_byte(aa);
		if(izb == INT_MAX) {
			*c1++ = cc;
		} else {
			ccpos = izb * 8;
			break;
		} 
	} while(izb == INT_MAX);
	/*
		for(int i = 0; i < 8; i++) {
			cc1 = (uint)(aa >> i * 8) & 0xffu;
			//ccpos += 8;
			if(cc1) {
				cc |= (uint64)cc1 << ccpos;
				ccpos += 8;
				if(ccpos == 64) {
					*c1++ = cc;
					ccpos = 0;
					//cc = aa >> i * 8;
					cc = 0;
				}
			} else
				break;
		}
	} while(cc1);
	*/
	// copy second string
	do {
		bb = *b1++;
		// commit current character group
		cc |= bb << ccpos;
		*c1++ = cc;
		// update for next
		izb = izero_byte(bb);
		cc = bb >> ccpos;
		if(izb != INT_MAX) {
			*c1++ = cc;
			break;
		}
	} while(izb == INT_MAX);

	/*
		for(int i = 0; i < 8; i++) {
			cc1 = (uint)(bb >> i * 8) & 0xffu;
			cc |= (uint64)cc1 << ccpos;
			ccpos += 8;
			if(ccpos == 64) {
				*c1++ = cc;
				ccpos = 0;
				//cc = bb >> i * 8;
				cc = 0;
			}
			if(!cc1)
				break;
		}
	} while(cc1);
	if(ccpos)
		*c1 = cc;
		*/
}  // dstrcat

/** adds two arrays of strings elementwise */
__global__ void add_strs_k
(char ** __restrict__ c, const char * const * __restrict__ a, 
 const char * const * __restrict__ b, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= n)
		return;
	// measure strings a and b
	const char *sa = a[i], *sb = b[i];
	int la = dstrlen(sa), lb = dstrlen(sb), lc = la + lb;
	//int la = 31, lb = 31, lc = la + lb;
	// allocate memory and get new string
	char *sc = (char *)hamalloc((lc + 1) * sizeof(char));
	dstrcat(sc, sa, sb);
	c[i] = sc;
}  // add_strs_k

void add_strs(char ** __restrict__ c, char **a, char **b, int n) {
#pragma omp parallel for
	for(int i = 0; i < n; i++) {
		const char *sa = a[i], *sb = b[i];
		int la = strlen(sa), lb = strlen(sb), lc = la + lb;
		//int la = 31, lb = 31, lc = la + lb;
		char *sc = (char *)malloc((lc + 1) * sizeof(char));
		strcpy(sc, sa);
		strcpy(sc + la, sb);
		c[i] = sc;
	}
}  // add_strs

#define MIN_LEN 31
#define MAX_LEN 31
#define PERIOD 32

/** a test for string addition on GPU */
void string_test_gpu(int n, bool print) {
	int min_len = MIN_LEN;
	int max_len = MAX_LEN;
	int period = PERIOD;
	// string lengths on host and device
	int *h_la = 0, *d_la = 0, *h_lb = 0, *d_lb = 0;
	size_t l_sz = n * sizeof(int), s_sz = n * sizeof(char *);
	cucheck(cudaMallocHost((void **)&h_la, l_sz));
	cucheck(cudaMallocHost((void **)&h_lb, l_sz));
	cucheck(cudaMalloc((void **)&d_la, l_sz));
	cucheck(cudaMalloc((void **)&d_lb, l_sz));
	random_array(h_la, n, period, min_len, max_len);
	random_array(h_lb, n, period, min_len, max_len);
	cucheck(cudaMemcpy(d_la, h_la, l_sz, cudaMemcpyHostToDevice));
	cucheck(cudaMemcpy(d_lb, h_lb, l_sz, cudaMemcpyHostToDevice));

	// string arrays
	char **d_sa, **d_sb, **d_sc;
	cucheck(cudaMalloc((void **)&d_sa, s_sz));
	cucheck(cudaMalloc((void **)&d_sb, s_sz));
	cucheck(cudaMalloc((void **)&d_sc, s_sz));

	// allocate strings
 	int bs = 128,	grid = divup(n, bs);
	double t1, t2;
	t1 = omp_get_wtime();
	alloc_strs_k<<<grid, bs>>>(d_sa, d_la, n);
	cucheck(cudaGetLastError());
	alloc_strs_k<<<grid, bs>>>(d_sb, d_lb, n);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	t2 = omp_get_wtime();
	//printf("t1 = %lf, t2 = %lf\n", t1, t2);
	if(print) {
		double t = (t2 - t1) / 2;
		printf("GPU allocation time: %4.2lf ms\n", t * 1e3);
		printf("GPU allocation performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	}

	//concatenate strings
	t1 = omp_get_wtime();
	add_strs_k<<<grid, bs>>>(d_sc, d_sa, d_sb, n);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	t2 = omp_get_wtime();
	if(print) {
		double t = t2 - t1;
		printf("GPU concatenation time: %4.2lf ms\n", t * 1e3);
		printf("GPU concatenation performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	}

	// free strings
	t1 = omp_get_wtime();
	free_strs_k<<<grid, bs>>>(d_sa, n);
	cucheck(cudaGetLastError());
	free_strs_k<<<grid, bs>>>(d_sb, n);
	cucheck(cudaGetLastError());
	free_strs_k<<<grid, bs>>>(d_sc, n);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	t2 = omp_get_wtime();
	if(print) {
		double t = (t2 - t1) / 3;
		//double t = (t2 - t1) / 2;
		printf("GPU freeing time: %4.2lf ms\n", t * 1e3);
		printf("GPU freeing performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	}

	// free the rest
	cucheck(cudaFree(d_sa));
	cucheck(cudaFree(d_sb));
	cucheck(cudaFree(d_sc));
	cucheck(cudaFree(d_la));
	cucheck(cudaFree(d_lb));
	cucheck(cudaFreeHost(h_la));
	cucheck(cudaFreeHost(h_lb));
}  // string_test_gpu

/** a test for string addition on CPU */
void string_test_cpu(int n, bool print) {
	int min_len = MIN_LEN;
	int max_len = MAX_LEN;
	int period = PERIOD;
	// string lengths on host and device
	int *h_la = 0, *h_lb = 0;
	size_t l_sz = n * sizeof(int), s_sz = n * sizeof(char *);
	h_la = (int *)malloc(l_sz);
	h_lb = (int *)malloc(l_sz);
	random_array(h_la, n, period, min_len, max_len);
	random_array(h_lb, n, period, min_len, max_len);

	// string arrays
	char **h_sa, **h_sb, **h_sc;
	h_sa = (char **)malloc(s_sz);
	h_sb = (char **)malloc(s_sz);
	h_sc = (char **)malloc(s_sz);

	// allocate strings
	double t1, t2;
	t1 = omp_get_wtime();
	alloc_strs(h_sa, h_la, n);
	alloc_strs(h_sb, h_lb, n);
	t2 = omp_get_wtime();
	//printf("t1 = %lf, t2 = %lf\n", t1, t2);
	if(print) {
		double t = (t2 - t1) / 2;
		printf("CPU allocation time: %4.2lf ms\n", t * 1e3);
		printf("CPU allocation performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	}

	//concatenate strings
	t1 = omp_get_wtime();
	add_strs(h_sc, h_sa, h_sb, n);
	t2 = omp_get_wtime();
	if(print) {
		double t = t2 - t1;
		printf("CPU concatenation time: %4.2lf ms\n", t * 1e3);
		printf("CPU concatenation performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	}

	// free strings
	t1 = omp_get_wtime();
	free_strs(h_sa, n);
	free_strs(h_sb, n);
	free_strs(h_sc, n);
	t2 = omp_get_wtime();
	if(print) {
		double t = (t2 - t1) / 3;
		//double t = (t2 - t1) / 2;
		printf("CPU freeing time: %4.2lf ms\n", t * 1e3);
		printf("CPU freeing performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	} 

	// free the rest
	free(h_sa);
	free(h_sb);
	free(h_sc);
	free(h_la);
	free(h_lb);
}  // string_test_cpu


int main(int argc, char **argv) {
	srandom((int)time(0));
	size_t memory = 512 * 1024 * 1024;
	// GPU test
	ha_init(halloc_opts_t(memory));
	//cucheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	string_test_gpu(10000, false);
	string_test_gpu(1000000, true);
	printf("==============================\n");
	// CPU test
	string_test_cpu(10000, false);
	string_test_cpu(1000000, true);
	ha_shutdown();
}  // main
