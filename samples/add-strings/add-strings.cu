/** @file grid-points.cu a test where grid points are sorted into a grid */

#define HALLOC_CPP
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

/** a random value in [a, b] range */
int random(int a, int b) {
	return a + random() % (b + 1);
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
	//if(i > n - 256)
	//	printf("i = %d, l = %d, n = %d\n", i, l, n);
	char *str = (char *)hamalloc((l + 1) * sizeof(char));
	// initialize
	for(int j = 0; j < l; j++)
		str[j] = '0' + j;
	str[l] = 0;
	// save string pointer
	strs[i] = str;
}  // alloc_strs_k

/** a kernel that frees memory allocated for strings */
__global__ void free_strs_k
(char ** __restrict__ strs, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= n)
		return;
	hafree(strs[i]);
}  // free_strs_k

// couple of helper device functions, analogous to C library
/** get the length of a string */
__device__ inline int dstrlen(const char * __restrict__ s) {
	int len = 0;
	while(*s++) len++;
	return len;
}  // strlen

/** concatenate two strings into the third string; all strings have been
		allocated, and the result has enough place to hold the arguments */
__device__ inline void dstrcat
(char *  __restrict__ c, const char * __restrict__ b, 
 const char * __restrict__ a) {
	while(*c++ = *a++) {}
	c--;
	while(*c++ = *b++) {}
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
	// allocate memory and get new string
	char *sc = (char *)hamalloc((lc + 1) * sizeof(char));
	dstrcat(sc, sa, sb);
	c[i] = sc;
	//if(i > n - 256)
	//	printf("c[%d][2] = %c\n", i, (int)sc[2]);
}  // add_strs_k

/** a test for string addition */
void string_test(int n, bool print) {
	int min_len = 127;
	int max_len = 127;
	int period = 32;
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
	int bs = 128, grid = divup(n, bs);
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
		printf("allocation time: %4.2lf ms\n", t * 1e3);
		printf("allocation performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	}
	
	// concatenate strings
	t1 = omp_get_wtime();
	add_strs_k<<<grid, bs>>>(d_sc, d_sa, d_sb, n);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	t2 = omp_get_wtime();
	if(print) {
		double t = t2 - t1;
		printf("concatenation time: %4.2lf ms\n", t * 1e3);
		printf("concatenation performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
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
		printf("freeing time: %4.2lf ms\n", t * 1e3);
		printf("freeing performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	}

	// free the rest
	cucheck(cudaFree(d_sa));
	cucheck(cudaFree(d_sb));
	cucheck(cudaFree(d_sc));
	cucheck(cudaFree(d_la));
	cucheck(cudaFree(d_lb));
	cucheck(cudaFreeHost(h_la));
	cucheck(cudaFreeHost(h_lb));
}  // string_test

int main(int argc, char **argv) {
	srandom((int)time(0));
	size_t memory = 512 * 1024 * 1024;
	//bool alloc = true;
	cucheck(cudaSetDevice(0));
	ha_init(halloc_opts_t(memory));
	// warm-up run
	string_test(10000, false);
	// main run
	string_test(200000, true);
	ha_shutdown();
}  // main
